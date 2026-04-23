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

import os

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

import numpy as np
import torch

from ..complex import ComplexTensor, complex_matmul

PMAX = 128  # Max partition dimension (systolic array rows)

# When set, dispatch bypasses torch_xla and runs kernels through
# `nki.simulate(kernel)(np_args)` on CPU. Lets us iterate kernels on any
# x86_64 Linux box without paying the NEFF compile + hardware dispatch
# cost. Semantics follow NKI 0.3.0's simulator: no NEFF compile, no
# SBUF/PSUM capacity checks, no latency/parallelism modelling. For
# correctness iteration only; hardware still owns perf numbers.
_USE_SIMULATOR = os.environ.get("TRNFFT_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)


def _use_simulator() -> bool:
    return _USE_SIMULATOR and HAS_NKI


_backend = "auto"


def set_backend(backend: str):
    """Set dispatch backend: 'auto', 'pytorch', or 'nki'."""
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires nki>=0.3.0 (Neuron SDK 2.29+)")
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


# complex_gemm_bf16 is defined later in the file (after the NKI kernels),
# because it references _complex_gemm_kernel_bf16 which only exists when HAS_NKI.


def complex_linear(x: ComplexTensor, w_real: torch.Tensor, w_imag: torch.Tensor) -> ComplexTensor:
    """Complex linear forward: y = x @ W^T (complex) with backend dispatch.

    NKI kernel fuses the 4 real matmuls and reuses the activation tile
    (loaded once, streamed against W_real and W_imag).
    """
    if _use_nki():
        return _nki_complex_linear(x, w_real, w_imag)
    # PyTorch fallback: 4 real matmuls
    yr = torch.matmul(x.real, w_real.t()) - torch.matmul(x.imag, w_imag.t())
    yi = torch.matmul(x.real, w_imag.t()) + torch.matmul(x.imag, w_real.t())
    return ComplexTensor(yr, yi)


def complex_mask_apply(mask: ComplexTensor, spec: ComplexTensor) -> ComplexTensor:
    """Apply complex mask to spectrogram with backend dispatch.

    NKI kernel fuses all 6 element-wise ops into single kernel invocation,
    avoiding 6 HBM round-trips.
    """
    if _use_nki():
        return _nki_complex_mask(mask, spec)
    return mask * spec


# --- NKI kernels ---
# Ported from neuron-complex-ops/kernels.py (Apache 2.0)

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
                    ar_t = nl.load_transpose2d(
                        a_real[m_off : m_off + TILE_M, k_off : k_off + TILE_K]
                    )
                    ai_t = nl.load_transpose2d(
                        a_imag[m_off : m_off + TILE_M, k_off : k_off + TILE_K]
                    )

                    # Load B col-tile with partition dim = K (already K-major).
                    br = nl.load(b_real[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    bi = nl.load(b_imag[k_off : k_off + TILE_K, n_off : n_off + TILE_N])

                    # NKI 2.24 doesn't support `psum -=` inside affine_range;
                    # use Vector Engine to negate B_imag, then accumulate with +=.
                    neg_bi = nl.negative(bi)

                    # C_real += A_real @ B_real  +  A_imag @ (-B_imag)
                    # NKI 0.3.0: nc_matmul is kwargs-only with dst= / accumulate=
                    # for in-place accumulation into the PSUM tile.
                    nisa.nc_matmul(dst=psum_cr, stationary=ar_t, moving=br, accumulate=True)
                    nisa.nc_matmul(dst=psum_cr, stationary=ai_t, moving=neg_bi, accumulate=True)

                    # C_imag += A_real @ B_imag  +  A_imag @ B_real
                    nisa.nc_matmul(dst=psum_ci, stationary=ar_t, moving=bi, accumulate=True)
                    nisa.nc_matmul(dst=psum_ci, stationary=ai_t, moving=br, accumulate=True)

                # NKI 0.3.0: nl.copy returns a view; PSUM→SBUF must go through
                # nisa.tensor_copy(dst=, src=) into a pre-allocated SBUF tile.
                # If the destination dtype differs from PSUM's fp32, cast after.
                cr_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                ci_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=cr_sbuf, src=psum_cr)
                nisa.tensor_copy(dst=ci_sbuf, src=psum_ci)
                if a_real.dtype != nl.float32:
                    cr_sbuf = nl.cast(cr_sbuf, dtype=a_real.dtype)
                    ci_sbuf = nl.cast(ci_sbuf, dtype=a_real.dtype)
                nl.store(c_real[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=cr_sbuf)
                nl.store(c_imag[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=ci_sbuf)

        return c_real, c_imag

    @nki.jit
    def _complex_gemm_kernel_bf16(a_real, a_imag, b_real, b_imag):
        """BF16 Complex GEMM: BF16 inputs → FP32 PSUM → FP32 output.

        Architectural pattern: nc_matmul accumulates into FP32 PSUM regardless
        of input dtype (hardware invariant on trn1). By allocating output as
        FP32 and *not* casting the PSUM back to BF16, we keep the full FP32
        accumulation quality while computing in BF16 (≈2× Tensor Engine throughput).

        This is the "PSUM is a free FP32 accumulator" principle:
          BF16 compute speed × FP32 accumulation precision.

        Inputs a_real, a_imag, b_real, b_imag are expected to be BF16.
        Output c_real, c_imag are always FP32 — the PSUM result, not rounded.

        The only difference from _complex_gemm_kernel: output dtype is nl.float32
        (not a_real.dtype) and no nl.cast is applied after nisa.tensor_copy.
        """
        M, K = a_real.shape
        _, N = b_real.shape

        TILE_M = min(M, 128)
        TILE_K = min(K, 128)
        TILE_N = min(N, 512)

        # Output is always FP32 — we keep the PSUM without rounding back to BF16.
        c_real = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
        c_imag = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum_cr = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                psum_ci = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    k_off = k * TILE_K

                    ar_t = nl.load_transpose2d(
                        a_real[m_off : m_off + TILE_M, k_off : k_off + TILE_K]
                    )
                    ai_t = nl.load_transpose2d(
                        a_imag[m_off : m_off + TILE_M, k_off : k_off + TILE_K]
                    )
                    br = nl.load(b_real[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    bi = nl.load(b_imag[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                    neg_bi = nl.negative(bi)

                    nisa.nc_matmul(dst=psum_cr, stationary=ar_t, moving=br, accumulate=True)
                    nisa.nc_matmul(dst=psum_cr, stationary=ai_t, moving=neg_bi, accumulate=True)
                    nisa.nc_matmul(dst=psum_ci, stationary=ar_t, moving=bi, accumulate=True)
                    nisa.nc_matmul(dst=psum_ci, stationary=ai_t, moving=br, accumulate=True)

                cr_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                ci_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=cr_sbuf, src=psum_cr)
                nisa.tensor_copy(dst=ci_sbuf, src=psum_ci)
                # No nl.cast — output remains FP32 from PSUM (the key difference).
                nl.store(c_real[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=cr_sbuf)
                nl.store(c_imag[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=ci_sbuf)

        return c_real, c_imag

    @nki.jit
    def _complex_linear_kernel(x_real, x_imag, w_real, w_imag):
        """Complex linear layer: y = x @ W^T where x and W are complex.

        Equivalent to:
            y_re = x_re @ W_re.T - x_im @ W_im.T
            y_im = x_re @ W_im.T + x_im @ W_re.T

        Shapes:
            x_real, x_imag: (M, K_in)         — activations
            w_real, w_imag: (K_out, K_in)     — weights (stored row-major)
            returns y_real, y_imag: (M, K_out)

        We treat this as a complex GEMM where W^T is the moving operand.
        Same calling convention as _complex_gemm_kernel: load x_re/x_im as
        stationary (transposed so partition dim = K_in), load w as moving.
        """
        M, K_in = x_real.shape
        K_out, _ = w_real.shape

        TILE_M = min(M, 128)
        TILE_K = min(K_in, 128)
        TILE_N = min(K_out, 512)

        y_real = nl.ndarray((M, K_out), dtype=x_real.dtype, buffer=nl.shared_hbm)
        y_imag = nl.ndarray((M, K_out), dtype=x_real.dtype, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(K_out // TILE_N):
                m_off = m * TILE_M
                n_off = n * TILE_N

                psum_yr = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                psum_yi = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K_in // TILE_K):
                    k_off = k * TILE_K

                    # Stationary: x tile, transposed so partition dim = K_in.
                    xr_t = nl.load_transpose2d(
                        x_real[m_off : m_off + TILE_M, k_off : k_off + TILE_K]
                    )
                    xi_t = nl.load_transpose2d(
                        x_imag[m_off : m_off + TILE_M, k_off : k_off + TILE_K]
                    )

                    # Moving: W^T columns, partition dim = K_in.
                    # W is (K_out, K_in) row-major; W^T is (K_in, K_out).
                    # Slice W[n_off:n_off+TILE_N, k_off:k_off+TILE_K] then transpose:
                    wr_t = nl.load_transpose2d(
                        w_real[n_off : n_off + TILE_N, k_off : k_off + TILE_K]
                    )
                    wi_t = nl.load_transpose2d(
                        w_imag[n_off : n_off + TILE_N, k_off : k_off + TILE_K]
                    )

                    # NKI 2.24 doesn't support `psum -=` in affine_range.
                    neg_wi_t = nl.negative(wi_t)

                    # y_real += x_real @ W_real^T  +  x_imag @ (-W_imag^T)
                    # NKI 0.3.0: kwargs-only in-place accumulation.
                    nisa.nc_matmul(dst=psum_yr, stationary=xr_t, moving=wr_t, accumulate=True)
                    nisa.nc_matmul(dst=psum_yr, stationary=xi_t, moving=neg_wi_t, accumulate=True)

                    # y_imag += x_real @ W_imag^T  +  x_imag @ W_real^T
                    nisa.nc_matmul(dst=psum_yi, stationary=xr_t, moving=wi_t, accumulate=True)
                    nisa.nc_matmul(dst=psum_yi, stationary=xi_t, moving=wr_t, accumulate=True)

                yr_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                yi_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=yr_sbuf, src=psum_yr)
                nisa.tensor_copy(dst=yi_sbuf, src=psum_yi)
                if x_real.dtype != nl.float32:
                    yr_sbuf = nl.cast(yr_sbuf, dtype=x_real.dtype)
                    yi_sbuf = nl.cast(yi_sbuf, dtype=x_real.dtype)
                nl.store(y_real[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=yr_sbuf)
                nl.store(y_imag[m_off : m_off + TILE_M, n_off : n_off + TILE_N], value=yi_sbuf)

        return y_real, y_imag

    @nki.jit
    def _complex_mul_kernel(a_real, a_imag, b_real, b_imag):
        """Fused element-wise complex multiply.

        Computes (a_re + i a_im) * (b_re + i b_im)
          = (a_re*b_re - a_im*b_im) + i (a_re*b_im + a_im*b_re)
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
        assert total % 128 == 0, (
            f"_complex_mul_kernel requires total size divisible by 128; got {total}"
        )

        free = total // 128
        # Free-dim tile size: cap at 512 to keep SBUF usage reasonable.
        # NKI affine_range can't evaluate min() symbolically, so use a constant
        # chunk size: the full free dim when it fits in one tile, or FMAX with
        # an exact-divisibility requirement. For power-of-2 input shapes, free
        # is always a power of 2 and FMAX=512 divides it exactly when free>FMAX.
        FMAX = 512
        if free <= FMAX:
            free_tile = free
        else:
            assert free % FMAX == 0, (
                f"_complex_mul_kernel: free={free} (total={total}) must be <= {FMAX} "
                f"or a multiple of {FMAX} to avoid dynamic tile sizing"
            )
            free_tile = FMAX

        c_real = nl.ndarray(shape, dtype=a_real.dtype, buffer=nl.shared_hbm)
        c_imag = nl.ndarray(shape, dtype=a_real.dtype, buffer=nl.shared_hbm)

        # Reshape views: flatten then expose as (128, free).
        a_re_2d = a_real.reshape((128, free))
        a_im_2d = a_imag.reshape((128, free))
        b_re_2d = b_real.reshape((128, free))
        b_im_2d = b_imag.reshape((128, free))
        c_re_2d = c_real.reshape((128, free))
        c_im_2d = c_imag.reshape((128, free))

        n_tiles = free // free_tile

        for t in nl.affine_range(n_tiles):
            f_off = t * free_tile
            f_end = f_off + free_tile  # constant slice size; no min()

            ar = nl.load(a_re_2d[:, f_off:f_end])
            ai = nl.load(a_im_2d[:, f_off:f_end])
            br = nl.load(b_re_2d[:, f_off:f_end])
            bi = nl.load(b_im_2d[:, f_off:f_end])

            # NKI 0.3.0: Python `*`/`+`/`-` operators aren't defined on
            # NkiTensor; use nl.multiply / nl.subtract / nl.add explicitly.
            cr = nl.subtract(nl.multiply(ar, br), nl.multiply(ai, bi))
            ci = nl.add(nl.multiply(ar, bi), nl.multiply(ai, br))

            nl.store(c_re_2d[:, f_off:f_end], value=cr)
            nl.store(c_im_2d[:, f_off:f_end], value=ci)

        return c_real, c_imag


# --- NKI kernel wrappers ---


def _to_xla(*tensors):
    """Move a list of tensors to the XLA device."""
    import torch_xla

    device = torch_xla.device()
    return [t.to(device) for t in tensors], tensors[0].device


def _simulate_kernel(kernel, *tensors):
    """Route a kernel call through `nki.simulate` on CPU.

    Converts each input tensor to numpy, calls the simulator, and marshals
    results back to torch tensors on the original device. Kernels with
    multiple HBM outputs return a tuple of numpy arrays; single-output
    kernels return a numpy array.

    Used by the dispatch wrappers when ``TRNFFT_USE_SIMULATOR=1`` is set
    in the environment. Bypasses torch_xla and NEFF compile entirely —
    correctness iteration only; hardware still owns perf numbers.
    """
    orig_device = tensors[0].device
    np_args = [t.detach().cpu().numpy() for t in tensors]
    out = nki.simulate(kernel)(*np_args)
    if isinstance(out, tuple):
        return tuple(torch.from_numpy(np.asarray(o)).to(orig_device) for o in out)
    return torch.from_numpy(np.asarray(out)).to(orig_device)


def _nki_complex_gemm(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """NKI complex GEMM with stationary reuse (autograd-aware)."""
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    from .autograd import complex_gemm_autograd

    c_real, c_imag = complex_gemm_autograd(a.real, a.imag, b.real, b.imag)
    return ComplexTensor(c_real, c_imag)


def complex_gemm_bf16(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """BF16 complex GEMM: BF16 inputs → FP32 PSUM → FP32 output.

    On NKI hardware: uses ``_complex_gemm_kernel_bf16`` which skips the
    PSUM→BF16 cast, preserving FP32 accumulation quality at BF16 throughput.
    On CPU (no NKI): falls back to ``complex_matmul`` with whatever dtype
    the inputs carry (correctness path; no throughput benefit on CPU).
    """
    if _use_nki() and not _use_simulator():
        import torch_xla

        device = torch_xla.device()
        a_re = a.real.to(device)
        a_im = a.imag.to(device)
        b_re = b.real.to(device)
        b_im = b.imag.to(device)
        c_real, c_imag = _complex_gemm_kernel_bf16(a_re, a_im, b_re, b_im)
        return ComplexTensor(c_real, c_imag)
    if _use_simulator():
        c_real, c_imag = _simulate_kernel(_complex_gemm_kernel_bf16, a.real, a.imag, b.real, b.imag)
        return ComplexTensor(c_real, c_imag)
    # CPU fallback: simulate FP32 PSUM behaviour by casting BF16 inputs to
    # FP32 before the matmul. On hardware, nc_matmul accumulates BF16 products
    # into a FP32 PSUM; on CPU torch.matmul accumulates in BF16 without the
    # FP32 PSUM, giving ~50% error at N=64. Casting to FP32 first matches the
    # hardware's FP32 accumulation quality at the cost of BF16 quantisation of
    # W and x (which is the intended accuracy regime: ~1e-3 rel error at N=256).
    a_fp32 = ComplexTensor(a.real.float(), a.imag.float())
    b_fp32 = ComplexTensor(b.real.float(), b.imag.float())
    result = complex_matmul(a_fp32, b_fp32)
    return ComplexTensor(result.real, result.imag)  # already FP32


def _ozaki_split_bf16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an FP32 tensor into high + low BF16 parts (Ozaki 1-level split).

    Returns ``(x_high, x_low)`` such that
    ``x_high.float() + x_low.float() ≈ x`` with error O(u_bf16^2 * |x|).

    x_high = bfloat16(x)                          — BF16 representation of x
    x_low  = bfloat16(x − float32(x_high))        — BF16 residual

    Used by :func:`complex_gemm_ozaki` to decompose matmul inputs into partial
    sums, enabling Ozaki-scheme FP64 emulation from BF16 kernel calls.
    """
    x_high = x.bfloat16()
    x_low = (x - x_high.float()).bfloat16()
    return x_high, x_low


def complex_gemm_ozaki(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """Ozaki-scheme complex GEMM: 1-level BF16 split + FP64 accumulation.

    Decomposes a and b into BF16 high/low parts and computes the three
    significant cross-terms using ``complex_gemm_bf16``, accumulating in FP64:

        3 BF16 matmuls: a_high@b_high + a_high@b_low + a_low@b_high

    The omitted term (a_low @ b_low) has magnitude O(u_bf16^2 * |a| * |b|)
    per entry, contributing O(sqrt(N) * u_bf16^2) relative error to the DFT
    output — ~1e-5 relative at N=64, ~2e-5 at N=256.

    Output dtype is torch.float32 (FP64 accumulation is internal only).
    Assumes a, b are FP32 on entry; they are split to BF16 internally.

    Note on multi-level Ozaki: a 2-level split (for O(u_bf16^4) accuracy)
    would require keeping the split residuals in FP32 rather than BF16, and
    is deferred to a future version.
    """
    a_r_h, a_r_l = _ozaki_split_bf16(a.real.float())
    a_i_h, a_i_l = _ozaki_split_bf16(a.imag.float())
    b_r_h, b_r_l = _ozaki_split_bf16(b.real.float())
    b_i_h, b_i_l = _ozaki_split_bf16(b.imag.float())

    def _gemm(ar, ai, br, bi):
        return complex_gemm_bf16(ComplexTensor(ar, ai), ComplexTensor(br, bi))

    # Three cross-terms: hh + hl + lh (omit ll which is O(u^2) × signal)
    hh = _gemm(a_r_h, a_i_h, b_r_h, b_i_h)
    hl = _gemm(a_r_h, a_i_h, b_r_l, b_i_l)
    lh = _gemm(a_r_l, a_i_l, b_r_h, b_i_h)

    # Accumulate in FP64 on CPU — Trainium XLA does not support FP64.
    # The three FP32 PSUM results are individually accurate; FP64 accumulation
    # prevents catastrophic cancellation when summing near-equal terms.
    def _cpu_f64(t: ComplexTensor) -> tuple[torch.Tensor, torch.Tensor]:
        return t.real.detach().cpu().double(), t.imag.detach().cpu().double()

    r_hh, i_hh = _cpu_f64(hh)
    r_hl, i_hl = _cpu_f64(hl)
    r_lh, i_lh = _cpu_f64(lh)
    out_r = (r_hh + r_hl + r_lh).float()
    out_i = (i_hh + i_hl + i_lh).float()
    return ComplexTensor(out_r, out_i)


def _nki_complex_linear(
    x: ComplexTensor, w_real: torch.Tensor, w_imag: torch.Tensor
) -> ComplexTensor:
    """NKI complex linear: y = x @ W^T (complex) via fused 4-matmul kernel (autograd-aware)."""
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    from .autograd import complex_linear_autograd

    y_real, y_imag = complex_linear_autograd(x.real, x.imag, w_real, w_imag)
    return ComplexTensor(y_real, y_imag)


def _nki_complex_mask(mask: ComplexTensor, spec: ComplexTensor) -> ComplexTensor:
    """NKI fused complex mask application (autograd-aware).

    The kernel requires total element count divisible by 128 (the Trainium
    Vector Engine partition limit). For inputs that aren't, fall back to
    PyTorch element-wise multiply (which is already autograd-safe).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    total = mask.real.numel()
    if total % 128 != 0:
        return mask * spec
    # Force contiguous layout before the kernel's reshape to (128, free).
    # Non-contiguous inputs (e.g., from broadcasting or transposed views) break
    # the reshape at NKI compile time even when the logical shape is fine.
    from .autograd import complex_mul_autograd

    c_real, c_imag = complex_mul_autograd(
        mask.real.contiguous(),
        mask.imag.contiguous(),
        spec.real.contiguous(),
        spec.imag.contiguous(),
    )
    return ComplexTensor(c_real, c_imag)
