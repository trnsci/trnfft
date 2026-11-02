"""Inputs the NKI kernels cannot take return or reshape on the host (#130).

On a Trainium host (``set_backend("nki")`` or ``"auto"``) a zero-size batch
reached the FFT kernels and failed lowering ("shape must be a tuple of
positive integers"), and ``ComplexLinear`` on a non-2-D input failed to
unpack its shape in the 2-D-only kernel. These tests force the NKI branch on
CPU and replace the kernels with stand-ins that refuse what the device
refuses.
"""

import pytest
import torch

import trnfft
import trnfft.fft_core as fc
import trnfft.nki.autograd as nki_autograd
import trnfft.nki.dispatch as dispatch
from trnfft import ComplexTensor
from trnfft.nn import ComplexLinear


@pytest.fixture
def nki_on(monkeypatch):
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    monkeypatch.setattr(fc, "_use_nki", lambda: True)


def _no_kernel(*args, **kwargs):
    raise AssertionError("shape must be a tuple of positive integers")


@pytest.mark.parametrize(
    "fn,shape",
    [
        (trnfft.fft, (0, 8)),
        (trnfft.ifft, (0, 8)),
        (trnfft.fft, (3, 0, 16)),
        (trnfft.fft, (0, 12)),  # Bluestein size
        (trnfft.rfft, (0, 8)),
        (trnfft.ihfft, (0, 8)),
        (trnfft.fft2, (0, 4, 4)),
        (trnfft.ifft2, (0, 4, 4)),
    ],
)
def test_zero_size_batch_skips_kernels(nki_on, monkeypatch, fn, shape):
    # The plan is built before any kernel launch; it must not be reached.
    monkeypatch.setattr(fc, "create_plan", _no_kernel)
    x = torch.randn(shape)
    got = fn(x)
    ref = getattr(torch.fft, fn.__name__)(x)
    assert tuple(got.shape) == tuple(ref.shape)
    assert got.real.numel() == 0


def _fake_linear(calls):
    def fake(xr, xi, wr, wi):
        calls.append(tuple(xr.shape))
        if xr.dim() != 2 or xr.numel() == 0:
            _no_kernel()
        return xr @ wr.t() - xi @ wi.t(), xr @ wi.t() + xi @ wr.t()

    return fake


@pytest.mark.parametrize("shape", [(6,), (2, 4, 6), (2, 3, 4, 6), (5, 6)])
def test_complex_linear_any_rank(nki_on, monkeypatch, shape):
    calls = []
    monkeypatch.setattr(nki_autograd, "complex_linear_autograd", _fake_linear(calls))
    torch.manual_seed(0)
    layer = ComplexLinear(6, 3)
    x = ComplexTensor(torch.randn(shape), torch.randn(shape))
    y = layer(x)
    monkeypatch.setattr(dispatch, "_use_nki", lambda: False)
    ref = layer(x)
    assert calls and all(len(c) == 2 for c in calls)
    assert y.shape == ref.shape == (*shape[:-1], 3)
    torch.testing.assert_close(y.real, ref.real)
    torch.testing.assert_close(y.imag, ref.imag)


@pytest.mark.parametrize("shape", [(0, 6), (2, 0, 6)])
def test_complex_linear_empty_batch(nki_on, monkeypatch, shape):
    monkeypatch.setattr(nki_autograd, "complex_linear_autograd", _no_kernel)
    layer = ComplexLinear(6, 3)
    y = layer(ComplexTensor(torch.randn(shape), torch.randn(shape)))
    assert y.shape == (*shape[:-1], 3)
