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

PMAX = 128  # Max partition dimension (systolic array rows)

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


# --- NKI kernels ---
# Ported from neuron-complex-ops/kernels.py (Apache 2.0, Playground Logic LLC)

if HAS_NKI:

    @nki.jit
    def _complex_gemm_kernel(a_real, a_imag, b_real, b_imag):
        """Complex GEMM via stationary tile reuse on Tensor Engine.

        Phase 1 — A_r stationary: PSUM_real += A_r.T @ B_r, PSUM_imag += A_r.T @ B_i
        Phase 2 — A_i stationary: PSUM_real += A_i.T @ (-B_i), PSUM_imag += A_i.T @ B_r
        """
        M, K = a_real.shape
        _, N = b_real.shape

        TILE_M = min(M, PMAX)
        TILE_K = min(K, PMAX)
        TILE_N = min(N, 512)

        c_real = nl.ndarray((M, N), dtype=a_real.dtype, buffer=nl.shared_hbm)
        c_imag = nl.ndarray((M, N), dtype=a_real.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum_cr = nl.zeros((TILE_M, TILE_N), dtype=nl.float32,
                                   buffer=nl.psum)
                psum_ci = nl.zeros((TILE_M, TILE_N), dtype=nl.float32,
                                   buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    ar = nl.load(a_real[m_off:m_off+TILE_M, k_off:k_off+TILE_K])
                    ai = nl.load(a_imag[m_off:m_off+TILE_M, k_off:k_off+TILE_K])
                    br = nl.load(b_real[k_off:k_off+TILE_K, n_off:n_off+TILE_N])
                    bi = nl.load(b_imag[k_off:k_off+TILE_K, n_off:n_off+TILE_N])

                    # Phase 1: A_r stationary
                    nisa.nc_matmul(psum_cr, ar, br)
                    nisa.nc_matmul(psum_ci, ar, bi)

                    # Vector Engine: negate B_i (overlaps with matmuls)
                    neg_bi = nl.negate(bi)

                    # Phase 2: A_i stationary
                    nisa.nc_matmul(psum_cr, ai, neg_bi)
                    nisa.nc_matmul(psum_ci, ai, br)

                cr_sbuf = nl.copy(psum_cr, dtype=a_real.dtype)
                ci_sbuf = nl.copy(psum_ci, dtype=a_real.dtype)
                nl.store(c_real[m_off:m_off+TILE_M, n_off:n_off+TILE_N], value=cr_sbuf)
                nl.store(c_imag[m_off:m_off+TILE_M, n_off:n_off+TILE_N], value=ci_sbuf)

        return c_real, c_imag

    @nki.jit
    def _complex_mul_kernel(a_real, a_imag, b_real, b_imag):
        """Fused element-wise complex multiply.

        Loads all 4 inputs in one pass, computes ac-bd and ad+bc in SBUF,
        writes 2 outputs. Avoids 6 separate HBM round-trips.
        """
        shape = a_real.shape
        total = 1
        for s in shape:
            total *= s

        TILE = min(total, PMAX * 512)

        c_real = nl.ndarray(shape, dtype=a_real.dtype, buffer=nl.shared_hbm)
        c_imag = nl.ndarray(shape, dtype=a_real.dtype, buffer=nl.shared_hbm)

        n_tiles = (total + TILE - 1) // TILE

        for t in nl.affine_range(n_tiles):
            off = t * TILE
            size = min(TILE, total - off)

            ar = nl.load(a_real.reshape((total,))[off:off+size])
            ai = nl.load(a_imag.reshape((total,))[off:off+size])
            br = nl.load(b_real.reshape((total,))[off:off+size])
            bi = nl.load(b_imag.reshape((total,))[off:off+size])

            cr = ar * br - ai * bi
            ci = ar * bi + ai * br

            nl.store(c_real.reshape((total,))[off:off+size], value=cr)
            nl.store(c_imag.reshape((total,))[off:off+size], value=ci)

        return c_real, c_imag


# --- NKI kernel wrappers ---

def _nki_complex_gemm(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """NKI complex GEMM with stationary reuse."""
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    c_real, c_imag = _complex_gemm_kernel(a.real, a.imag, b.real, b.imag)
    return ComplexTensor(c_real, c_imag)


def _nki_complex_mask(mask: ComplexTensor, spec: ComplexTensor) -> ComplexTensor:
    """NKI fused complex mask application."""
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    c_real, c_imag = _complex_mul_kernel(mask.real, mask.imag, spec.real, spec.imag)
    return ComplexTensor(c_real, c_imag)
