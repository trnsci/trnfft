"""
trnfft — FFT and complex-valued operations for AWS Trainium via NKI.

Drop-in replacement for torch.fft on Trainium hardware.
Includes complex neural network layers for speech/audio/physics workloads.
Part of the trnsci scientific computing suite.

Incorporates neuron-complex-ops (ComplexTensor, NKI dispatch, complex NN layers).
"""

__version__ = "0.26.0"

from .api import (
    fft,
    fft2,
    fftfreq,
    fftn,
    fftshift,
    hfft,
    hfft2,
    hfftn,
    ifft,
    ifft2,
    ifftn,
    ifftshift,
    ihfft,
    ihfft2,
    ihfftn,
    irfft,
    irfft2,
    irfftn,
    istft,
    rfft,
    rfft2,
    rfftfreq,
    rfftn,
    stft,
)
from .complex import ComplexTensor, complex_matmul
from .fft_core import set_ozaki_product_precision_verified
from .nki import HAS_NKI, get_backend, set_backend
from .plan import FFTPlan, clear_plan_cache, create_plan
from .precision import get_precision, set_precision

__all__ = [
    # FFT operations
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "hfft2",
    "ihfft2",
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "hfftn",
    "ihfftn",
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
    "stft",
    "istft",
    # Complex tensor
    "ComplexTensor",
    "complex_matmul",
    # Plan management
    "create_plan",
    "clear_plan_cache",
    "FFTPlan",
    # Backend
    "HAS_NKI",
    "set_backend",
    "get_backend",
    # Precision
    "set_precision",
    "set_ozaki_product_precision_verified",
    "get_precision",
]
