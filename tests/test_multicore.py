"""Tests for multi-NeuronCore FFT dispatch (trnfft/nki/multicore.py).

CPU-runnable tests validate:
  - Batch split produces identical output to single-core
  - Shape contracts: (batch, n) in → (batch, n) out
  - Core count clamping (batch < num_cores → uses batch count)
  - API surface: set_multicore, get_multicore, multi_core_fft

Hardware-only tests (marked @pytest.mark.neuron) will be added once
torch_neuronx.DataParallel dispatch is validated on trn1/trn2.
"""

import numpy as np
import pytest
import torch

from trnfft.complex import ComplexTensor
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

    def test_multi_core_fft_single_transform_raises(self):
        """Single 1-D input raises NotImplementedError when multicore enabled."""
        set_multicore(True)
        x = ComplexTensor(torch.randn(64), torch.randn(64))
        with pytest.raises(NotImplementedError, match="Stage parallelism"):
            multi_core_fft(x)


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
