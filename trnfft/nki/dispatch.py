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
        """Complex GEMM: C = A @ B where A, B, C are complex.

        Uses 4 real matmuls accumulated into 2 PSUM tiles:
            C_real = A_real @ B_real - A_imag @ B_imag
            C_imag = A_real @ B_imag + A_imag @ B_real

        NKI 2.24 calling convention:
            psum[...] += nisa.nc_matmul(stationary, moving)
        Both inputs must be SBUF tiles. Stationary partition ≤ 128 (= K).
        Stationary free ≤ 128 (= M). Moving free ≤ 512 (= N).

        Tile shapes (after reshaping for the systolic array):
            stationary (A row-tile, transposed): (TILE_K, TILE_M)
            moving (B col-tile):                 (TILE_K, TILE_N)
            result (PSUM):                       (TILE_M, TILE_N)
        """
        M, K = a_real.shape
        _, N = b_real.shape

        TILE_M = min(M, 128)
        TILE_K = min(K, 128)
        TILE_N = min(N, 512)

        c_real = nl.ndarray((M, N), dtype=a_real.dtype, buffer=nl.shared_hbm)
        c_imag = nl.ndarray((M, N), dtype=a_real.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum_cr = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                psum_ci = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    # Load A row-tile and transpose so partition dim = K.
                    # nl.load gives (TILE_M, TILE_K); need (TILE_K, TILE_M).
                    ar_t = nl.load_transpose2d(a_real[m_off:m_off+TILE_M, k_off:k_off+TILE_K])
                    ai_t = nl.load_transpose2d(a_imag[m_off:m_off+TILE_M, k_off:k_off+TILE_K])

                    # Load B col-tile with partition dim = K (already K-major).
                    br = nl.load(b_real[k_off:k_off+TILE_K, n_off:n_off+TILE_N])
                    bi = nl.load(b_imag[k_off:k_off+TILE_K, n_off:n_off+TILE_N])

                    # C_real += A_real @ B_real  -  A_imag @ B_imag
                    psum_cr[...] += nisa.nc_matmul(ar_t, br)
                    psum_cr[...] -= nisa.nc_matmul(ai_t, bi)

                    # C_imag += A_real @ B_imag  +  A_imag @ B_real
                    psum_ci[...] += nisa.nc_matmul(ar_t, bi)
                    psum_ci[...] += nisa.nc_matmul(ai_t, br)

                cr_sbuf = nl.copy(psum_cr, dtype=a_real.dtype)
                ci_sbuf = nl.copy(psum_ci, dtype=a_real.dtype)
                nl.store(c_real[m_off:m_off+TILE_M, n_off:n_off+TILE_N], value=cr_sbuf)
                nl.store(c_imag[m_off:m_off+TILE_M, n_off:n_off+TILE_N], value=ci_sbuf)

        return c_real, c_imag

    @nki.jit
    def _complex_mul_kernel(a_real, a_imag, b_real, b_imag):
        """Fused element-wise complex multiply.

        Computes (a_re + i a_im) * (b_re + i b_im) = (a_re*b_re - a_im*b_im) + i(a_re*b_im + a_im*b_re)
        in a single kernel, avoiding 6 separate HBM round-trips.

        NKI 2.24 partition-dim constraint: any SBUF tile must be 2D with first
        dim being the partition dim, and partition size ≤ 128. We reshape the
        flat input into (PMAX, ceil(total/PMAX)) and process row-by-column tiles.
        Inputs must have total size divisible by PMAX (128).
        """
        shape = a_real.shape
        total = 1
        for s in shape:
            total *= s

        # Trainium's Vector Engine partition limit. Inputs that are not multiples
        # of 128 elements would need a tail tile — caller pads if necessary.
        assert total % 128 == 0, \
            f"_complex_mul_kernel requires total size divisible by 128; got {total}"

        free = total // 128
        # Free-dim tile size: cap at 512 to keep SBUF usage reasonable.
        FMAX = 512
        free_tile = min(free, FMAX)

        c_real = nl.ndarray(shape, dtype=a_real.dtype, buffer=nl.shared_hbm)
        c_imag = nl.ndarray(shape, dtype=a_real.dtype, buffer=nl.shared_hbm)

        # Reshape views: flatten then expose as (128, free).
        a_re_2d = a_real.reshape((128, free))
        a_im_2d = a_imag.reshape((128, free))
        b_re_2d = b_real.reshape((128, free))
        b_im_2d = b_imag.reshape((128, free))
        c_re_2d = c_real.reshape((128, free))
        c_im_2d = c_imag.reshape((128, free))

        n_tiles = (free + free_tile - 1) // free_tile

        for t in nl.affine_range(n_tiles):
            f_off = t * free_tile
            f_end = min(f_off + free_tile, free)

            ar = nl.load(a_re_2d[:, f_off:f_end])
            ai = nl.load(a_im_2d[:, f_off:f_end])
            br = nl.load(b_re_2d[:, f_off:f_end])
            bi = nl.load(b_im_2d[:, f_off:f_end])

            cr = ar * br - ai * bi
            ci = ar * bi + ai * br

            nl.store(c_re_2d[:, f_off:f_end], value=cr)
            nl.store(c_im_2d[:, f_off:f_end], value=ci)

        return c_real, c_imag


# --- NKI kernel wrappers ---

def _nki_complex_gemm(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """NKI complex GEMM with stationary reuse."""
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    c_real, c_imag = _complex_gemm_kernel(a.real, a.imag, b.real, b.imag)
    return ComplexTensor(c_real, c_imag)


def _nki_complex_mask(mask: ComplexTensor, spec: ComplexTensor) -> ComplexTensor:
    """NKI fused complex mask application.

    The kernel requires total element count divisible by 128 (the Trainium
    Vector Engine partition limit). For inputs that aren't, fall back to
    PyTorch element-wise multiply.
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    total = mask.real.numel()
    if total % 128 != 0:
        return mask * spec
    c_real, c_imag = _complex_mul_kernel(mask.real, mask.imag, spec.real, spec.imag)
    return ComplexTensor(c_real, c_imag)
