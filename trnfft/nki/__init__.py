"""NKI kernel implementations and dispatch for Trainium."""

from .dispatch import (
    HAS_NKI,
    complex_gemm,
    complex_linear,
    complex_mask_apply,
    get_backend,
    set_backend,
)
from .multicore import get_multicore, multi_core_fft, set_multicore

if HAS_NKI:
    from .butterfly import butterfly_stage_kernel
else:
    butterfly_stage_kernel = None

__all__ = [
    "HAS_NKI",
    "set_backend",
    "get_backend",
    "complex_gemm",
    "complex_mask_apply",
    "complex_linear",
    "butterfly_stage_kernel",
    "set_multicore",
    "get_multicore",
    "multi_core_fft",
]
