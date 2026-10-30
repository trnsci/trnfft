"""
Native TorchNeuron dispatch on hardware (Trn2/Trn3, torch-neuronx 2.14+).

Skipped unless the dispatch path resolves to "native": these do not run on
the XLA lineage (Trn1) or off-hardware. The existing hardware suites
(TestNKIFFT, TestLargeFFTNeuron, TestNKIGradients, ...) also exercise the
native path automatically on a TorchNeuron host, since they pass CPU tensors
through the dispatch layer; this file covers what is specific to it:
custom-op registration, no torch_xla, each op against the NKI simulator,
register_autograd, device-resident inputs, and torch.compile.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

import trnfft
from trnfft import ComplexTensor
from trnfft.nki.dispatch import HAS_NKI, _get_dispatch_path, _use_simulator


def _native() -> bool:
    return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]


def _rand(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype)


def _ops():
    from trnfft.nki.dispatch import _get_native_ops

    return _get_native_ops()


def _simulate(kernel, *args):
    """CPU reference: the same @nki.jit kernel under nki.simulate."""
    import nki

    np_args = [
        (a.float() if a.dtype == torch.bfloat16 else a).numpy()
        if isinstance(a, torch.Tensor)
        else a
        for a in args
    ]
    out = nki.simulate(kernel)(*np_args)
    return tuple(torch.from_numpy(np.asarray(o)) for o in out)


def _dev(*tensors):
    return [t.to("neuron") for t in tensors]


def _assert_pair_close(got, ref, rtol=1e-5, atol=1e-5):
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g.cpu().float(), r.float(), rtol=rtol, atol=atol)


def _np_complex(y: ComplexTensor) -> np.ndarray:
    return y.real.detach().cpu().double().numpy() + 1j * y.imag.detach().cpu().double().numpy()


def _rel(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


class TestNativeDispatch:
    def test_native_path_never_imports_torch_xla(self):
        """Path selection and op registration in a fresh interpreter never
        import torch_xla.

        The subprocess deliberately stops before any kernel launch: inside
        the suite the parent pytest already holds the visible NeuronCore, so
        a launch in a child process fails in ``nrt_init``. The launch half of
        the check runs in-process in ``test_launch_does_not_import_torch_xla``.
        """
        code = (
            "import sys, torch, trnfft; "
            "from trnfft.nki import dispatch as d; "
            "assert d._dispatch_path is None; "
            "assert d._native_available(), 'no native backend'; "
            "d._dispatch_path = 'native'; "
            "d._get_native_ops(); "
            "assert 'trnfft.nki._native' in sys.modules; "
            "assert hasattr(torch.ops.trnfft, 'complex_gemm'); "
            "assert hasattr(torch.ops.trnfft, 'fft'); "
            "assert 'torch_xla' not in sys.modules, 'torch_xla imported'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_launch_does_not_import_torch_xla(self, nki_backend):
        if "torch_xla" in sys.modules:
            pytest.skip("torch_xla was imported by another test in this process")
        x = ComplexTensor(_rand(4, 256, seed=1), _rand(4, 256, seed=2))
        y = trnfft.fft(x)
        assert _rel(_np_complex(y), np.fft.fft(_np_complex(x))) < 1e-5
        assert "trnfft.nki._native" in sys.modules, "kernel did not launch"
        assert "torch_xla" not in sys.modules

    def test_ops_registered_in_torch_ops(self):
        _ops()
        for name in (
            "complex_gemm",
            "complex_gemm_bf16",
            "complex_linear",
            "complex_mul",
            "butterfly_stage",
            "butterfly_stage_kahan",
            "stockham_radix4_stage",
            "stockham_radix8_w8",
            "four_step_fft",
            "fft",
        ):
            assert hasattr(torch.ops.trnfft, name), name


class TestNativeOpsEager:
    """Each registered op on device vs the same kernel under nki.simulate."""

    def test_complex_gemm(self):
        from trnfft.nki.dispatch import _complex_gemm_kernel

        args = [_rand(128, 128, seed=i) for i in range(4)]
        out = torch.ops.trnfft.complex_gemm(*_dev(*args))
        assert out[0].device.type == "neuron"
        _assert_pair_close(out, _simulate(_complex_gemm_kernel, *args), rtol=1e-4, atol=1e-4)

    def test_complex_gemm_bf16(self):
        from trnfft.nki.dispatch import _complex_gemm_kernel_bf16

        args = [_rand(128, 128, seed=i, dtype=torch.bfloat16) for i in range(4)]
        out = torch.ops.trnfft.complex_gemm_bf16(*_dev(*args))
        assert out[0].dtype == torch.float32
        _assert_pair_close(out, _simulate(_complex_gemm_kernel_bf16, *args), rtol=1e-3, atol=1e-3)

    def test_complex_linear(self):
        from trnfft.nki.dispatch import _complex_linear_kernel

        args = [_rand(128, 128, seed=i) for i in range(4)]
        out = torch.ops.trnfft.complex_linear(*_dev(*args))
        _assert_pair_close(out, _simulate(_complex_linear_kernel, *args), rtol=1e-4, atol=1e-4)

    def test_complex_mul(self):
        from trnfft.nki.dispatch import _complex_mul_kernel

        args = [_rand(128, 1024, seed=i) for i in range(4)]
        out = torch.ops.trnfft.complex_mul(*_dev(*args))
        _assert_pair_close(out, _simulate(_complex_mul_kernel, *args))

    @pytest.mark.parametrize("kahan", [False, True])
    @pytest.mark.parametrize("stage", [0, 3])
    def test_butterfly_stage(self, kahan, stage):
        from trnfft.nki import butterfly

        b, n = 8, 16
        m = 1 << (stage + 1)
        rows, half = b * (n // m), m >> 1
        args = [_rand(b, n, seed=1), _rand(b, n, seed=2)]
        args += [_rand(rows, half, seed=3), _rand(rows, half, seed=4)]
        kernel = (
            butterfly.butterfly_stage_kernel_kahan if kahan else butterfly.butterfly_stage_kernel
        )
        op = torch.ops.trnfft.butterfly_stage_kahan if kahan else torch.ops.trnfft.butterfly_stage
        out = op(*_dev(*args), n, stage)
        _assert_pair_close(out, _simulate(kernel, *args, n, stage))

    @pytest.mark.parametrize("rows", [64, 256])
    def test_stockham_radix4_stage(self, rows):
        from trnfft.nki.stockham import stockham_radix4_stage_kernel

        args = [_rand(rows, 4, seed=i) for i in range(4)]
        out = torch.ops.trnfft.stockham_radix4_stage(*_dev(*args))
        _assert_pair_close(out, _simulate(stockham_radix4_stage_kernel, *args))

    def test_stockham_radix8_w8(self):
        from trnfft.nki.stockham import stockham_radix8_w8_kernel

        args = [
            _rand(64, 8, seed=1),
            _rand(64, 8, seed=2),
            _rand(8, 8, seed=3),
            _rand(8, 8, seed=4),
        ]
        out = torch.ops.trnfft.stockham_radix8_w8(*_dev(*args))
        _assert_pair_close(out, _simulate(stockham_radix8_w8_kernel, *args), rtol=1e-4, atol=1e-4)


class TestNativeFourStepFused:
    """Batch-fused four-step kernel on device (trnfft#82 follow-up)."""

    def test_op_matches_simulator(self):
        from trnfft import fft_core as fc
        from trnfft.nki.four_step import four_step_fft_kernel

        consts = fc._four_step_fused_consts(64, 64, False, torch.float32, torch.device("cpu"), 2, 8)
        args = [_rand(38, 4096, seed=1), _rand(38, 4096, seed=2), *consts]
        out = torch.ops.trnfft.four_step_fft(*_dev(*args))
        assert out[0].device.type == "neuron"
        _assert_pair_close(out, _simulate(four_step_fft_kernel, *args), rtol=1e-4, atol=1e-3)

    # Ragged batches exercise P < 128 // N2 (odd B) and the tail group.
    @pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
    @pytest.mark.parametrize("batch", [1, 3, 38, 1024])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_fft_matches_numpy(self, nki_backend, n, batch, inverse):
        from trnfft import fft_core as fc

        x = ComplexTensor(_rand(batch, n, seed=n + batch), _rand(batch, n, seed=n - batch))
        xc = _np_complex(x)
        ref = np.fft.ifft(xc) if inverse else np.fft.fft(xc)
        x = ComplexTensor(x.real.to("neuron"), x.imag.to("neuron"))
        assert fc._use_fused_four_step(x, n)
        y = (trnfft.ifft if inverse else trnfft.fft)(x)
        assert y.real.device.type == "neuron"
        # FP32 four-step on Trn2 measures 1.2e-7 to 1.6e-7.
        assert _rel(_np_complex(y), ref) < 5e-7

    def test_row_chunks_match_one_launch(self, nki_backend, monkeypatch):
        from trnfft import fft_core as fc

        x = ComplexTensor(
            _rand(40, 4096, seed=11).to("neuron"), _rand(40, 4096, seed=12).to("neuron")
        )
        whole = trnfft.fft(x)
        monkeypatch.setattr(fc, "_FOUR_STEP_FUSED_MAX_ROWS", 16)
        chunked = trnfft.fft(x)
        assert torch.equal(whole.real.cpu(), chunked.real.cpu())
        assert torch.equal(whole.imag.cpu(), chunked.imag.cpu())


class TestNativeFFT:
    """End to end through trnfft.fft on the native path."""

    # 256: single GEMM; 2048/4096: four-step; 2^14/2^16: Stockham (mixed / r4).
    @pytest.mark.parametrize("n", [256, 2048, 4096, 1 << 14, 1 << 16])
    @pytest.mark.parametrize("on_device", [False, True])
    def test_fft_matches_numpy(self, nki_backend, n, on_device):
        x = ComplexTensor(_rand(4, n, seed=1), _rand(4, n, seed=2))
        ref = np.fft.fft(_np_complex(x))
        if on_device:
            x = ComplexTensor(x.real.to("neuron"), x.imag.to("neuron"))
        y = trnfft.fft(x)
        assert y.real.device.type == ("neuron" if on_device else "cpu")
        assert _rel(_np_complex(y), ref) < 1e-5

    @pytest.mark.parametrize("n", [7, 1000])
    def test_bluestein_on_device(self, nki_backend, n):
        x = ComplexTensor(_rand(n, seed=3).to("neuron"), _rand(n, seed=4).to("neuron"))
        y = trnfft.fft(x)
        assert y.real.device.type == "neuron"
        assert _rel(_np_complex(y), np.fft.fft(_np_complex(x))) < 1e-5

    def test_kahan_butterfly_path(self, nki_backend, monkeypatch):
        from trnfft import fft_core as fc

        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 0)  # force the Kahan butterfly
        x = ComplexTensor(_rand(4, 1024, seed=5), _rand(4, 1024, seed=6))
        y = fc.fft_core(x, precision="kahan")
        assert _rel(_np_complex(y), np.fft.fft(_np_complex(x))) < 1e-5

    def test_fft_backward_register_autograd(self, nki_backend):
        x_re = _rand(4, 256, seed=7).requires_grad_(True)
        x_im = _rand(4, 256, seed=8).requires_grad_(True)
        y = trnfft.fft(ComplexTensor(x_re, x_im))
        w_re, w_im = _rand(4, 256, seed=9), _rand(4, 256, seed=10)
        (y.real * w_re + y.imag * w_im).sum().backward()

        xc = torch.complex(x_re.detach().double(), x_im.detach().double()).requires_grad_(True)
        ref = torch.fft.fft(xc)
        (ref.real * w_re.double() + ref.imag * w_im.double()).sum().backward()
        torch.testing.assert_close(x_re.grad.double(), xc.grad.real, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(x_im.grad.double(), xc.grad.imag, rtol=1e-4, atol=1e-4)

    def test_complex_linear_backward_on_device(self, nki_backend):
        from trnfft.nki.autograd import complex_linear_autograd

        xs = [_rand(128, 128, seed=i).to("neuron").requires_grad_(True) for i in range(4)]
        y_re, y_im = complex_linear_autograd(*xs)
        (y_re.square().sum() + y_im.square().sum()).backward()
        assert all(t.grad is not None and t.grad.device.type == "neuron" for t in xs)

        ref = [t.detach().cpu().double().requires_grad_(True) for t in xs]
        xr, xi, wr, wi = ref
        r_re = xr @ wr.t() - xi @ wi.t()
        r_im = xr @ wi.t() + xi @ wr.t()
        (r_re.square().sum() + r_im.square().sum()).backward()
        for t, r in zip(xs, ref, strict=True):
            torch.testing.assert_close(t.grad.cpu().double(), r.grad, rtol=1e-3, atol=1e-2)


class TestNativeTorchCompile:
    """nki_op ops inside torch.compile(backend="neuron", fullgraph=True)."""

    def test_complex_gemm_compiled(self):
        op = _ops()["complex_gemm"]

        @torch.compile(backend="neuron", fullgraph=True)
        def f(a, b, c, d):
            re, im = op(a, b, c, d)
            return re * 2.0, im * 2.0

        args = [_rand(128, 128, seed=i) for i in range(4)]
        eager = torch.ops.trnfft.complex_gemm(*_dev(*args))
        out = f(*_dev(*args))
        _assert_pair_close(out, tuple(e.cpu() * 2.0 for e in eager), rtol=1e-4, atol=1e-4)

    def test_complex_mul_compiled(self):
        op = _ops()["complex_mul"]

        @torch.compile(backend="neuron", fullgraph=True)
        def f(a, b, c, d):
            return op(a, b, c, d)

        args = [_rand(128, 512, seed=i) for i in range(4)]
        ar, ai, br, bi = args
        _assert_pair_close(f(*_dev(*args)), (ar * br - ai * bi, ar * bi + ai * br))

    def test_stockham_radix4_stage_compiled(self):
        from trnfft.nki.stockham import stockham_radix4_stage_kernel

        op = _ops()["stockham_radix4_stage"]

        @torch.compile(backend="neuron", fullgraph=True)
        def f(a, b, c, d):
            return op(a, b, c, d)

        args = [_rand(256, 4, seed=i) for i in range(4)]
        _assert_pair_close(f(*_dev(*args)), _simulate(stockham_radix4_stage_kernel, *args))

    @pytest.mark.xfail(
        strict=False,
        reason="trnfft::fft is a host composite (several launches, host reshapes); "
        "whether the neuron backend accepts it as an opaque op is untested",
    )
    def test_fft_composite_compiled(self):
        op = _ops()["fft"]

        @torch.compile(backend="neuron", fullgraph=True)
        def f(re, im):
            return op(re, im, False, "fast")

        x = ComplexTensor(_rand(4, 256, seed=1), _rand(4, 256, seed=2))
        y = f(x.real.to("neuron"), x.imag.to("neuron"))
        assert _rel(_np_complex(ComplexTensor(*y)), np.fft.fft(_np_complex(x))) < 1e-5
