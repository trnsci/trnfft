"""NKI kernel implementations and dispatch for Trainium."""

from .dispatch import HAS_NKI, set_backend, get_backend, complex_gemm, complex_mask_apply

if HAS_NKI:
    from .butterfly import butterfly_stage_kernel
else:
    butterfly_stage_kernel = None

__all__ = [
    "HAS_NKI", "set_backend", "get_backend",
    "complex_gemm", "complex_mask_apply",
    "butterfly_stage_kernel",
]
