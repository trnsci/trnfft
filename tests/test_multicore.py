"""Tests for multi-NeuronCore FFT dispatch (trnfft/nki/multicore.py).

CPU-runnable tests validate:
  - Batch split produces identical output to single-core
  - Shape contracts: (batch, n) in → (batch, n) out
  - Core count clamping (batch < num_cores → uses batch count)
  - API surface: set_multicore, get_multicore, multi_core_fft

The CPU tests run the sequential CPU split, also on a Neuron host: the
autouse ``cpu_split_path`` fixture hides torch_neuronx from the module, so
they never depend on how many NeuronCores are visible (trnfft#159).
``TestVisibleCoreClamp`` fakes the native path with a chosen visible count.

Hardware-only tests are marked ``@pytest.mark.neuron``.
"""

import numpy as np
import pytest
import torch

from trnfft.complex import ComplexTensor
from trnfft.nki import dispatch
from trnfft.nki import multicore as mc
from trnfft.nki.multicore import (
    _batch_split_fft,
    _factorize,
    _resolve_num_cores,
    _stage_parallel_fft,
    get_multicore,
    multi_core_fft,
    set_multicore,
)


@pytest.fixture(autouse=True)
def reset_multicore():
    """Restore multicore state after each test."""
    was_enabled = get_multicore()
    yield
    set_multicore(was_enabled)


@pytest.fixture(autouse=True)
def cpu_split_path(request, monkeypatch):
    """CPU tests take the sequential CPU split, whatever the host.

    On a Neuron host ``HAS_TORCH_NEURONX`` is True and the split would go to
    the visible NeuronCores, so the results (and ``_resolve_num_cores``)
    would depend on ``NEURON_RT_VISIBLE_CORES``. Neuron-marked tests keep
    the real path.
    """
    if "neuron" not in request.keywords:
        monkeypatch.setattr(mc, "HAS_TORCH_NEURONX", False)


class TestMulticoreAPI:
    def test_set_get_multicore(self):
        set_multicore(True)
        assert get_multicore() is True
        set_multicore(False)
        assert get_multicore() is False

    def test_set_multicore_with_num_cores(self):
        from trnfft.nki.multicore import _num_cores

        set_multicore(True, num_cores=4)
        from trnfft.nki import multicore

        assert multicore._num_cores == 4
        set_multicore(False, num_cores=0)
        assert multicore._num_cores == 0

    def test_multi_core_fft_disabled_passthrough(self):
        """When disabled, multi_core_fft calls single-core path."""
        set_multicore(False)
        n = 64
        x = ComplexTensor(torch.randn(4, n), torch.randn(4, n))
        result = multi_core_fft(x)
        assert result.real.shape == (4, n)
        assert result.imag.shape == (4, n)

    def test_multi_core_fft_single_transform_prime_raises(self):
        """Single 1-D prime-N input raises NotImplementedError when multicore enabled."""
        set_multicore(True)
        x = ComplexTensor(torch.randn(7), torch.randn(7))
        with pytest.raises(NotImplementedError, match="prime"):
            multi_core_fft(x)

    def test_multi_core_fft_single_transform_composite_succeeds(self):
        """Single 1-D composite-N input succeeds via stage-parallel path."""
        set_multicore(True)
        n = 64  # 8 × 8
        ct = ComplexTensor(torch.randn(n), torch.zeros(n))
        result = multi_core_fft(ct)
        assert result.real.shape == (n,)


class TestBatchSplitCPU:
    """CPU correctness tests: batch split output must match single-core output."""

    @pytest.mark.parametrize(
        "batch,n,cores",
        [
            (4, 64, 2),
            (8, 128, 4),
            (6, 256, 3),
            (3, 64, 2),  # batch not divisible by cores → last shard smaller
        ],
    )
    def test_batch_split_matches_single_core(self, batch, n, cores):
        torch.manual_seed(42)
        real = torch.randn(batch, n)
        imag = torch.randn(batch, n)
        x = ComplexTensor(real, imag)

        from trnfft.fft_core import fft_core

        expected = fft_core(x)
        actual = _batch_split_fft(x, inverse=False, num_cores=cores)

        np.testing.assert_allclose(
            actual.real.numpy(),
            expected.real.numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"batch={batch} n={n} cores={cores}: real mismatch",
        )
        np.testing.assert_allclose(
            actual.imag.numpy(),
            expected.imag.numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"batch={batch} n={n} cores={cores}: imag mismatch",
        )

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_batch_split_inverse(self, n):
        """FFT → IFFT roundtrip via batch-split path recovers original."""
        torch.manual_seed(7)
        batch = 4
        real = torch.randn(batch, n)
        imag = torch.zeros(batch, n)
        x = ComplexTensor(real, imag)

        X = _batch_split_fft(x, inverse=False, num_cores=2)
        back = _batch_split_fft(X, inverse=True, num_cores=2)

        np.testing.assert_allclose(
            back.real.numpy(),
            real.numpy(),
            atol=1e-4,
            err_msg=f"N={n}: roundtrip real mismatch",
        )
        np.testing.assert_allclose(
            back.imag.numpy(),
            imag.numpy(),
            atol=1e-4,
            err_msg=f"N={n}: roundtrip imag mismatch",
        )

    def test_batch_split_core_clamp(self):
        """When batch < num_cores, actual_cores is clamped to batch size."""
        batch, n = 2, 64
        torch.manual_seed(0)
        x = ComplexTensor(torch.randn(batch, n), torch.zeros(batch, n))
        # Request 8 cores but only 2 samples — should not error.
        result = _batch_split_fft(x, inverse=False, num_cores=8)
        assert result.real.shape == (batch, n)

    def test_batch_split_single_core_passthrough(self):
        """num_cores=1 processes without splitting."""
        n = 64
        torch.manual_seed(1)
        x = ComplexTensor(torch.randn(4, n), torch.zeros(4, n))
        result = _batch_split_fft(x, inverse=False, num_cores=1)
        assert result.real.shape == (4, n)

    def test_output_dtype_is_fp32(self):
        n = 64
        x = ComplexTensor(torch.randn(4, n), torch.zeros(4, n))
        result = _batch_split_fft(x, inverse=False, num_cores=2)
        assert result.real.dtype == torch.float32
        assert result.imag.dtype == torch.float32

    def test_output_shape_preserved(self):
        for batch, n in [(1, 64), (8, 256), (16, 128)]:
            x = ComplexTensor(torch.randn(batch, n), torch.zeros(batch, n))
            result = _batch_split_fft(x, inverse=False, num_cores=2)
            assert result.real.shape == (batch, n), f"batch={batch} n={n}: shape mismatch"


class TestResolveNumCores:
    def test_explicit_cores_clamped_to_batch(self):
        from trnfft.nki import multicore

        multicore._num_cores = 8
        assert _resolve_num_cores(batch_size=4) == 4
        assert _resolve_num_cores(batch_size=16) == 8
        multicore._num_cores = 0

    def test_zero_cores_falls_back_to_cpu_default(self):
        from trnfft.nki import multicore

        multicore._num_cores = 0
        # On CPU (no torch_neuronx), falls back to min(2, batch)
        assert _resolve_num_cores(batch_size=1) == 1
        assert _resolve_num_cores(batch_size=4) == 2
        assert _resolve_num_cores(batch_size=2) == 2


class TestFactorize:
    @pytest.mark.parametrize(
        "n,expected_product",
        [(4, 4), (64, 64), (256, 256), (1024, 1024), (65536, 65536)],
    )
    def test_factorize_product_is_n(self, n, expected_product):
        n1, n2 = _factorize(n)
        assert n1 * n2 == expected_product

    def test_factorize_n1_leq_sqrt_n(self):
        import math

        for n in [4, 16, 64, 256, 1024]:
            n1, n2 = _factorize(n)
            assert n1 <= math.sqrt(n) + 1
            assert n1 >= 1

    def test_factorize_prime_raises(self):
        with pytest.raises(ValueError, match="prime"):
            _factorize(65537)

    def test_factorize_prime_small(self):
        with pytest.raises(ValueError):
            _factorize(7)


class TestStageParallelFFT:
    """CPU correctness tests for row-column stage-parallel FFT."""

    @pytest.mark.parametrize("n", [4, 16, 64, 256, 1024])
    def test_stage_parallel_matches_single_core(self, n):
        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))

        from trnfft.fft_core import fft_core

        expected = fft_core(ct)
        actual = _stage_parallel_fft(ct, num_cores=2, inverse=False)

        np.testing.assert_allclose(
            actual.real.numpy(),
            expected.real.numpy(),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"N={n}: real mismatch",
        )
        np.testing.assert_allclose(
            actual.imag.numpy(),
            expected.imag.numpy(),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"N={n}: imag mismatch",
        )

    @pytest.mark.parametrize("n", [64, 256, 1024])
    def test_stage_parallel_roundtrip(self, n):
        torch.manual_seed(7)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))

        X = _stage_parallel_fft(ct, num_cores=2, inverse=False)
        back = _stage_parallel_fft(X, num_cores=2, inverse=True)

        np.testing.assert_allclose(
            back.real.numpy(), x.numpy(), atol=1e-4, err_msg=f"N={n}: roundtrip mismatch"
        )

    def test_stage_parallel_output_shape(self):
        for n in [4, 64, 256]:
            ct = ComplexTensor(torch.randn(n), torch.zeros(n))
            result = _stage_parallel_fft(ct, num_cores=2, inverse=False)
            assert result.real.shape == (n,), f"N={n}: shape {result.real.shape}"

    def test_stage_parallel_output_dtype_fp32(self):
        ct = ComplexTensor(torch.randn(64), torch.zeros(64))
        result = _stage_parallel_fft(ct, num_cores=2, inverse=False)
        assert result.real.dtype == torch.float32
        assert result.imag.dtype == torch.float32


class TestMultiCoreSingleTransform:
    """multi_core_fft single-transform routing tests."""

    def test_single_transform_composite_n(self):
        """Composite N routes to stage-parallel, produces correct output."""
        set_multicore(True)
        n = 64
        torch.manual_seed(42)
        ct = ComplexTensor(torch.randn(n), torch.zeros(n))

        from trnfft.fft_core import fft_core

        expected = fft_core(ct)
        actual = multi_core_fft(ct)

        np.testing.assert_allclose(actual.real.numpy(), expected.real.numpy(), rtol=1e-4, atol=1e-4)

    def test_single_transform_prime_n_raises(self):
        """Prime N raises NotImplementedError with helpful message."""
        set_multicore(True)
        ct = ComplexTensor(torch.randn(7), torch.zeros(7))
        with pytest.raises(NotImplementedError, match="prime"):
            multi_core_fft(ct)


class TestSixStepMulticore:
    """Host-staged six-step split across cores (trnfft#54); CPU devices stand in."""

    @pytest.mark.parametrize("batch,n,cores", [(1, 256, 3), (2, 512, 2), (1, 8192, 4)])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_matches_numpy(self, batch, n, cores, inverse):
        from trnfft.nki.multicore import _six_step_multicore

        g = torch.Generator().manual_seed(n + cores)
        xr, xi = torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g)
        y = _six_step_multicore(
            ComplexTensor(xr, xi), inverse, cores, [torch.device("cpu")] * cores
        )
        xc = xr.double().numpy() + 1j * xi.double().numpy()
        ref = np.fft.ifft(xc) if inverse else np.fft.fft(xc)
        got = y.real.double().numpy() + 1j * y.imag.double().numpy()
        assert y.shape == (batch, n)
        assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-5

    @pytest.mark.parametrize(
        "total,parts,expected",
        [
            (8, 4, [(0, 2), (2, 4), (4, 6), (6, 8)]),
            (3, 4, [(0, 1), (1, 2), (2, 3)]),
            (5, 1, [(0, 5)]),
        ],
    )
    def test_split_range(self, total, parts, expected):
        from trnfft.nki.multicore import _split_range

        assert _split_range(total, parts) == expected


@pytest.fixture
def visible(monkeypatch):
    """Fake native host path with ``visible(k)`` logical cores (CPU devices stand in)."""

    def make(k: int) -> None:
        monkeypatch.setattr(mc, "HAS_TORCH_NEURONX", True)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(dispatch, "_neuron_host_allocator", lambda: False)
        monkeypatch.setattr(
            mc, "_core_devices", lambda src, n: [torch.device("cpu")] * max(1, min(n, k))
        )
        mc._WARM.clear()
        mc._CLAMP_WARNED.clear()

    return make


def _rel_to_torch(y, xr, xi, inverse=False):
    xc = torch.complex(xr.double(), xi.double())
    ref = torch.fft.ifft(xc) if inverse else torch.fft.fft(xc)
    got = torch.complex(y.real.double(), y.imag.double())
    return (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()


class TestVisibleCoreClamp:
    """More cores requested than visible (trnfft#159): clamp with one warning.

    Before the fix the native host split made one shard per requested core
    but only ran the shards that had a visible core, so the other rows of
    the output were never written.
    """

    @pytest.mark.parametrize("batch,cores,vis", [(8, 4, 2), (6, 4, 3), (5, 8, 2), (4, 4, 1)])
    def test_batch_split_clamped(self, visible, batch, cores, vis):
        visible(vis)
        g = torch.Generator().manual_seed(batch * 10 + cores)
        xr, xi = torch.randn(batch, 64, generator=g), torch.randn(batch, 64, generator=g)
        with pytest.warns(UserWarning, match=f"{cores} NeuronCores requested but {vis} visible"):
            y = _batch_split_fft(ComplexTensor(xr, xi), inverse=False, num_cores=cores)
        assert y.real.shape == (batch, 64)
        assert _rel_to_torch(y, xr, xi) < 1e-5

    def test_warns_once(self, visible, recwarn):
        visible(2)
        x = ComplexTensor(torch.randn(8, 64), torch.randn(8, 64))
        for _ in range(3):
            _batch_split_fft(x, inverse=False, num_cores=4)
        assert sum("NeuronCores requested" in str(w.message) for w in recwarn.list) == 1

    def test_within_visible_no_warning(self, visible, recwarn):
        visible(4)
        x = ComplexTensor(torch.randn(8, 64), torch.randn(8, 64))
        _batch_split_fft(x, inverse=False, num_cores=4)
        assert not any("NeuronCores requested" in str(w.message) for w in recwarn.list)

    @pytest.mark.parametrize("inverse", [False, True])
    def test_stage_parallel_clamped(self, visible, inverse):
        visible(2)
        g = torch.Generator().manual_seed(159)
        xr, xi = torch.randn(1024, generator=g), torch.randn(1024, generator=g)
        with pytest.warns(UserWarning, match="NeuronCores requested"):
            y = _stage_parallel_fft(ComplexTensor(xr, xi), num_cores=8, inverse=inverse)
        assert _rel_to_torch(y, xr, xi, inverse) < 1e-5

    def test_dispatch_rejects_more_shards_than_cores(self, visible):
        visible(2)
        shards = torch.randn(6, 64).chunk(3)
        with pytest.raises(ValueError, match="3 shards for 2 visible"):
            mc._neuron_dp_dispatch(shards, shards, 64, False, 3)

    def test_batch_within_visible_no_warning(self, visible, recwarn):
        """8 requested, 2 visible, but a batch of 2 needs only 2 shards."""
        visible(2)
        x = ComplexTensor(torch.randn(2, 64), torch.randn(2, 64))
        _batch_split_fft(x, inverse=False, num_cores=8)
        assert not any("NeuronCores requested" in str(w.message) for w in recwarn.list)


@pytest.mark.neuron
class TestVisibleCoreClampNeuron:
    """On hardware: request more cores than ``NEURON_RT_VISIBLE_CORES`` exposes
    and check host-operand results against torch.fft (trnfft#159)."""

    @pytest.mark.parametrize("batch,n", [(16, 256), (7, 1024), (64, 4096)])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_batch_split_over_visible(self, nki_backend, batch, n, inverse):
        import torch_neuronx

        vis = torch_neuronx.device_count()
        cores = vis + 3
        g = torch.Generator().manual_seed(batch + n)
        xr, xi = torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g)
        mc._CLAMP_WARNED.clear()
        for _ in range(2):  # first call runs the shards in turn, then concurrently
            with pytest.warns(UserWarning, match=f"{cores} NeuronCores requested but {vis}"):
                mc._CLAMP_WARNED.clear()
                y = _batch_split_fft(ComplexTensor(xr, xi), inverse=inverse, num_cores=cores)
            assert y.real.device.type == "cpu" and y.real.shape == (batch, n)
            assert _rel_to_torch(y, xr, xi, inverse) < 1e-5

    @pytest.mark.parametrize("n", [1024, 4096])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_stage_parallel_over_visible(self, nki_backend, n, inverse):
        import torch_neuronx

        cores = torch_neuronx.device_count() + 3
        g = torch.Generator().manual_seed(n)
        xr, xi = torch.randn(n, generator=g), torch.randn(n, generator=g)
        mc._CLAMP_WARNED.clear()
        with pytest.warns(UserWarning, match="NeuronCores requested"):
            y = _stage_parallel_fft(ComplexTensor(xr, xi), num_cores=cores, inverse=inverse)
        assert y.real.device.type == "cpu"
        assert _rel_to_torch(y, xr, xi, inverse) < 1e-5

    @pytest.mark.parametrize("cores", [2, 3, 4])
    def test_batch_split_within_visible(self, nki_backend, cores):
        """k <= visible on host operands: every shard on its own core, no warning."""
        import warnings

        import torch_neuronx

        if torch_neuronx.device_count() < cores:
            pytest.skip(f"needs >= {cores} visible NeuronCores (NEURON_RT_VISIBLE_CORES)")
        g = torch.Generator().manual_seed(cores)
        xr, xi = torch.randn(37, 1024, generator=g), torch.randn(37, 1024, generator=g)
        for inverse in (False, True):
            for _ in range(2):
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", message=".*NeuronCores requested.*")
                    y = _batch_split_fft(ComplexTensor(xr, xi), inverse=inverse, num_cores=cores)
                assert _rel_to_torch(y, xr, xi, inverse) < 1e-5
            y = _stage_parallel_fft(ComplexTensor(xr[0], xi[0]), num_cores=cores, inverse=inverse)
            assert _rel_to_torch(y, xr[0], xi[0], inverse) < 1e-5
