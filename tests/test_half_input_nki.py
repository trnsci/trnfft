"""float16 / bfloat16 input on the NKI path runs the fp32 kernels (#133).

On Trn2, half-precision real input to fft / ifft / rfft / ihfft / stft failed
to lower (``AttributeError: module 'nki.language' has no attribute 'cast'``)
on ``set_backend("nki")`` and ``"auto"``. The transform now runs on fp32
copies and returns the input dtype, as torch.fft does. These tests force the
NKI branch on CPU and replace the kernel entry with a stand-in that refuses
anything but float32.
"""

import pathlib

import pytest
import torch

import trnfft
import trnfft.fft_core as fc
import trnfft.nki.autograd as nki_autograd
import trnfft.nki.dispatch as dispatch
from trnfft import ComplexTensor

HALF = [torch.float16, torch.bfloat16]
TOL = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}


@pytest.fixture
def nki_on(monkeypatch):
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    monkeypatch.setattr(fc, "_use_nki", lambda: True)


def _fp32_only_kernel(seen):
    def fake(xr, xi, inverse, precision):
        seen.append(xr.dtype)
        if xr.dtype != torch.float32 or xi.dtype != torch.float32:
            raise AssertionError(f"kernel got {xr.dtype}/{xi.dtype}: lowering fails on Trn2")
        z = torch.complex(xr.double(), xi.double())
        y = torch.fft.ifft(z) if inverse else torch.fft.fft(z)
        return y.real.float(), y.imag.float()

    return fake


def _c128(y):
    return torch.complex(y.real.double(), y.imag.double())


def _rel(got, want):
    return ((_c128(got) - want).abs().max() / want.abs().max()).item()


@pytest.mark.parametrize("dtype", HALF, ids=str)
@pytest.mark.parametrize("n", [12, 64, 129, 256])
@pytest.mark.parametrize("op", ["fft", "ifft", "rfft", "ihfft"])
def test_half_real_input_runs_fp32_kernel(nki_on, monkeypatch, op, n, dtype):
    seen = []
    monkeypatch.setattr(nki_autograd, "fft_autograd", _fp32_only_kernel(seen))
    x = (torch.randn(3, n, generator=torch.Generator().manual_seed(n)) * 5).to(dtype)
    y = getattr(trnfft, op)(x)
    assert seen and set(seen) == {torch.float32}
    assert y.real.dtype == dtype and y.imag.dtype == dtype
    want = getattr(torch.fft, op)(x.double())
    assert tuple(y.real.shape) == tuple(want.shape)
    # One rounding of the result to the input dtype, no half arithmetic.
    assert _rel(y, want) <= TOL[dtype]


@pytest.mark.parametrize("dtype", HALF, ids=str)
def test_half_complex_input_runs_fp32_kernel(nki_on, monkeypatch, dtype):
    seen = []
    monkeypatch.setattr(nki_autograd, "fft_autograd", _fp32_only_kernel(seen))
    g = torch.Generator().manual_seed(1)
    xr, xi = torch.randn(4, 100, generator=g).to(dtype), torch.randn(4, 100, generator=g).to(dtype)
    y = trnfft.fft(ComplexTensor(xr, xi))
    assert set(seen) == {torch.float32}
    assert y.real.dtype == dtype
    want = torch.fft.fft(torch.complex(xr.double(), xi.double()))
    assert _rel(y, want) <= TOL[dtype]


@pytest.mark.parametrize("dtype", HALF, ids=str)
def test_half_stft_runs_fp32_kernel(nki_on, monkeypatch, dtype):
    seen = []
    monkeypatch.setattr(nki_autograd, "fft_autograd", _fp32_only_kernel(seen))
    x = torch.randn(2, 128, generator=torch.Generator().manual_seed(7)).to(dtype)
    w = torch.hann_window(32, dtype=dtype)
    y = trnfft.stft(x, 32, hop_length=8, window=w)
    assert seen and set(seen) == {torch.float32}
    assert y.real.dtype == dtype
    want = torch.stft(x.double(), 32, hop_length=8, window=w.double(), return_complex=True)
    assert _rel(y, want) <= TOL[dtype]


@pytest.mark.parametrize("dtype", HALF, ids=str)
def test_half_past_lowp_range_runs_fp32(monkeypatch, dtype):
    """A low-precision mode past its GEMM range reaches the general driver,
    which upcasts half input before any fp32 kernel path."""
    seen = []

    def fake_gemm(x, inverse):
        seen.append(x.real.dtype)
        assert x.real.dtype == torch.float32
        y = torch.fft.fft(torch.complex(x.real.double(), x.imag.double()))
        return ComplexTensor(y.real.float(), y.imag.float())

    import trnfft.nki.butterfly as bf

    # The driver imports the butterfly kernels up front; they only exist with nki.
    for name in ("butterfly_stage_kernel", "butterfly_stage_kernel_kahan"):
        monkeypatch.setattr(bf, name, None, raising=False)
    monkeypatch.setattr(fc, "_fft_via_gemm", fake_gemm)
    monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1 << 20)
    monkeypatch.setattr(fc, "_use_fused_four_step", lambda x, n: False)
    n = fc._LOWP_GEMM_MAX * 2
    x = torch.randn(2, n, generator=torch.Generator().manual_seed(3)).to(dtype)
    y = fc._cooley_tukey_nki_nograd(ComplexTensor(x, torch.zeros_like(x)), False, "bf16")
    assert seen == [torch.float32]
    assert y.real.dtype == dtype
    assert _rel(y, torch.fft.fft(x.double())) <= TOL[dtype]


def test_half_input_off_nki_path_untouched(monkeypatch):
    """The pytorch backend keeps its own half handling (no upcast wrapper)."""
    called = []
    monkeypatch.setattr(fc, "_half_via_fp32", lambda *a: called.append(1))
    monkeypatch.setattr(fc, "_use_nki", lambda: False)
    y = trnfft.fft(torch.randn(2, 16).half())
    assert not called and y.real.dtype == torch.float16


def test_no_nl_cast_in_kernels():
    """``nl.cast`` does not exist in NKI 0.7: every half-output kernel failed
    to lower with AttributeError (#133). PSUM -> SBUF converts in tensor_copy."""
    root = pathlib.Path(trnfft.__file__).parent / "nki"
    hits = [
        f"{p.name}:{i}"
        for p in sorted(root.glob("*.py"))
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if "nl.cast(" in line
    ]
    assert not hits, hits
