"""
trnfft — FFT and complex-valued operations for AWS Trainium via NKI.

Drop-in replacement for torch.fft on Trainium hardware.
Includes complex neural network layers for speech/audio/physics workloads.
Part of the trnsci scientific computing suite.

Incorporates neuron-complex-ops (ComplexTensor, NKI dispatch, complex NN layers).
"""

__version__ = "0.11.0"

from .api import (
    fft, ifft, rfft, irfft,
    fft2, fftn, ifftn,
    rfft2, irfft2, rfftn, irfftn,
    stft, istft,
)
from .complex import ComplexTensor, complex_matmul
from .plan import create_plan, clear_plan_cache, FFTPlan
from .nki import HAS_NKI, set_backend, get_backend
from .precision import set_precision, get_precision

__all__ = [
    # FFT operations
    "fft", "ifft", "rfft", "irfft",
    "fft2", "fftn", "ifftn",
    "rfft2", "irfft2", "rfftn", "irfftn",
    "stft", "istft",
    # Complex tensor
    "ComplexTensor", "complex_matmul",
    # Plan management
    "create_plan", "clear_plan_cache", "FFTPlan",
    # Backend
    "HAS_NKI", "set_backend", "get_backend",
    # Precision
    "set_precision", "get_precision",
]
