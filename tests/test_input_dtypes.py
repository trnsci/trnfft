"""Integer / bool inputs are promoted to floating point, as torch.fft does (#113)."""

import pytest
import torch

import trnfft
from trnfft import ComplexTensor


def _c(ct):
    return torch.complex(ct.real.double(), ct.imag.double())


INT_DTYPES = [torch.int16, torch.int32, torch.int64, torch.uint8]


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("n", [8, 12, 64, 100])
@pytest.mark.parametrize(
    "op, ref",
    [
        (trnfft.fft, torch.fft.fft),
        (trnfft.ifft, torch.fft.ifft),
        (trnfft.rfft, torch.fft.rfft),
        (trnfft.ihfft, torch.fft.ihfft),
        (trnfft.fft2, torch.fft.fft2),
        (trnfft.rfft2, torch.fft.rfft2),
        (trnfft.fftn, torch.fft.fftn),
        (trnfft.rfftn, torch.fft.rfftn),
    ],
)
def test_integer_input_matches_torch(op, ref, n, dtype):
    g = torch.Generator().manual_seed(n)
    x = torch.randint(0, 50, (3, n), generator=g).to(dtype)
    y = op(x)
    expected = ref(x)
    assert y.dtype == torch.get_default_dtype()
    torch.testing.assert_close(_c(y), expected.to(torch.complex128), rtol=1e-4, atol=1e-3)


def test_bool_input_matches_torch():
    x = torch.tensor([[True, False, True, True, False, False, True, False, True, True]])
    torch.testing.assert_close(
        _c(trnfft.fft(x)), torch.fft.fft(x).to(torch.complex128), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("n", [8, 10])
def test_integer_complextensor_irfft(n):
    re = torch.arange(n // 2 + 1).reshape(1, -1)
    im = torch.zeros_like(re)
    y = trnfft.irfft(ComplexTensor(re, im), n=n)
    assert y.dtype == torch.get_default_dtype()
    expected = torch.fft.irfft(torch.complex(re.double(), im.double()), n=n)
    torch.testing.assert_close(y.double(), expected, rtol=1e-4, atol=1e-5)


def test_float_inputs_untouched():
    x = torch.randn(2, 16, dtype=torch.float64)
    assert trnfft.fft(x).dtype == torch.float64
    assert trnfft.rfft(x.half()).dtype == torch.float16
