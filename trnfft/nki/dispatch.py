"""
NKI dispatch layer — auto-selects between PyTorch and NKI backends.

Folded from neuron-complex-ops. When running on Trainium with neuronxcc
available, dispatches to optimized NKI kernels. Otherwise uses PyTorch ops.

Backend selection:
    "auto"    — NKI if available, else PyTorch
    "pytorch" — Always PyTorch (works on CPU, CUDA, Trainium via TorchNeuron)
    "nki"     — Always NKI (fails if neuronxcc not installed)
"""

from __future__ import annotations

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

from ..complex import ComplexTensor, complex_matmul
import torch

_backend = "auto"


def set_backend(backend: str):
    """Set dispatch backend: 'auto', 'pytorch', or 'nki'."""
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires neuronxcc. Install with: pip install neuronxcc")
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


def complex_gemm(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """Complex matrix multiply with backend dispatch.

    NKI kernel uses stationary tile reuse:
    - Phase 1: A_real stationary, stream B_real and B_imag
    - Phase 2: A_imag stationary, stream -B_imag and B_real
    - 4 SBUF loads instead of 8, PSUM accumulation
    """
    if _use_nki():
        return _nki_complex_gemm(a, b)
    return complex_matmul(a, b)


def complex_mask_apply(mask: ComplexTensor, spec: ComplexTensor) -> ComplexTensor:
    """Apply complex mask to spectrogram with backend dispatch.

    NKI kernel fuses all 6 element-wise ops into single kernel invocation,
    avoiding 6 HBM round-trips.
    """
    if _use_nki():
        return _nki_complex_mask(mask, spec)
    return mask * spec


# --- NKI kernel wrappers (stubs until validated on hardware) ---

def _nki_complex_gemm(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """NKI complex GEMM with stationary reuse.

    TODO: Wire to actual NKI kernel once validated on trn1/trn2.
    See neuron-complex-ops kernels_optimized.py for the reference implementation.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    # Fallback to PyTorch until kernel is validated
    return complex_matmul(a, b)


def _nki_complex_mask(mask: ComplexTensor, spec: ComplexTensor) -> ComplexTensor:
    """NKI fused complex mask application.

    TODO: Wire to actual NKI kernel once validated on trn1/trn2.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    return mask * spec
