"""
FFT core algorithms.

Iterative radix-2 Cooley-Tukey (decimation-in-time) for power-of-2 sizes.
Bluestein chirp-z for arbitrary sizes.
All operations on split real/imaginary tensors for Trainium compatibility.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .complex import ComplexTensor
from .nki.dispatch import _use_nki
from .plan import FFTAlgorithm, FFTPlan, create_plan


def fft_core(
    x: ComplexTensor,
    inverse: bool = False,
    plan: FFTPlan | None = None,
    precision: str | None = None,
) -> ComplexTensor:
    """Compute 1-D FFT along last dimension.

    If plan is None, one is created (and cached) automatically.

    ``precision`` selects the numerical mode: ``None`` uses the global
    (see ``trnfft.set_precision``); otherwise one of ``"fast"`` / ``"kahan"`` /
    ``"double"``.
    """
    from .precision import _resolve

    prec = _resolve(precision)

    n = x.shape[-1]
    if n == 1:
        return x.clone()

    if plan is None:
        plan = create_plan(n, inverse=inverse)

    if plan.algorithm == FFTAlgorithm.COOLEY_TUKEY:
        return _cooley_tukey(x, inverse, precision=prec)
    else:
        return _bluestein(x, inverse, padded_n=plan.padded_n, precision=prec)


def _cooley_tukey(x: ComplexTensor, inverse: bool, precision: str = "fast") -> ComplexTensor:
    """Iterative radix-2 decimation-in-time FFT.

    Standard textbook algorithm:
    1. Bit-reversal permutation
    2. log2(N) butterfly stages
    3. For inverse: divide by N at the end

    Dispatches to NKI butterfly kernel when running on Trainium hardware.
    The ``precision`` mode selects the butterfly variant when NKI is active;
    host-side CPU path is already FP32 and isn't affected by "kahan".
    """
    if _use_nki():
        return _cooley_tukey_nki(x, inverse, precision=precision)

    n = x.shape[-1]
    log2n = int(math.log2(n))
    assert 1 << log2n == n, f"Not power of 2: {n}"

    sign = 1.0 if inverse else -1.0

    # Bit-reversal permutation
    indices = _bit_reverse_indices(n, log2n)
    re = x.real[..., indices].clone()
    im = x.imag[..., indices].clone()

    # Precompute twiddle factors for all stages
    # W_N^k = exp(sign * 2πi * k / N)
    for s in range(log2n):
        m = 1 << (s + 1)  # Butterfly group size
        half = m >> 1  # Half group

        # Twiddle: exp(sign * 2πi * k / m) for k = 0..half-1
        angles = sign * 2.0 * math.pi * torch.arange(half, dtype=re.dtype) / m
        tw_re = torch.cos(angles)
        tw_im = torch.sin(angles)

        for k in range(half):
            # All butterflies at position k within their group, across all groups
            # Even indices: k, k+m, k+2m, ...
            # Odd indices: k+half, k+half+m, k+half+2m, ...
            even_idx = list(range(k, n, m))
            odd_idx = list(range(k + half, n, m))

            e_re = re[..., even_idx]
            e_im = im[..., even_idx]
            o_re = re[..., odd_idx]
            o_im = im[..., odd_idx]

            # Complex multiply: twiddle * odd
            t_re = tw_re[k]
            t_im = tw_im[k]
            prod_re = t_re * o_re - t_im * o_im
            prod_im = t_re * o_im + t_im * o_re

            # Butterfly
            re[..., even_idx] = e_re + prod_re
            im[..., even_idx] = e_im + prod_im
            re[..., odd_idx] = e_re - prod_re
            im[..., odd_idx] = e_im - prod_im

    result = ComplexTensor(re, im)
    if inverse:
        result = result * (1.0 / n)
    return result


# DFT-as-GEMM fast path. Rationale:
#   The Trainium Tensor engine (`nisa.nc_matmul`) executes a full matmul with
#   PSUM accumulation at throughput that dwarfs the Vector engine, which is
#   what every butterfly stage bottlenecks on. For small N, replacing
#   log2(N) butterfly stages with one `W @ x` matmul (O(N^2) work but on the
#   fast engine, with one HBM round-trip instead of log2(N)) is a
#   straight win — asymmetric hardware affordances flip the usual
#   complexity-vs-constant tradeoff.
#
#   Threshold set to 256: measured upper bound where FP32 O(N^2)
#   nc_matmul accumulation stays within 1e-3 relative error vs numpy
#   reference. Widened bench on trn1 (2026-04-13, docs/design-notes/
#   fft-is-a-gemm.md) shows the launch-count win extends cleanly to
#   N=1024 (5.3× over butterfly), with a hard perf cliff at N=2048.
#   The 256 cap is precision-bound, not perf-bound — raising it further
#   needs Stockham radix-r (Thread B) to break the O(N^2) accumulation
#   without falling back to butterfly. test_fft_nki_vs_numpy confirmed
#   N=256 stays inside 1e-3 tol; N=1024 exceeds (observed 2.2% rel err).
_DFT_GEMM_THRESHOLD = 256

# Upper bound for the FP64 CPU DFT-GEMM path (precision="double" only).
# Beyond this N, "double" mode falls through to NKI Stockham (~1e-4 FP32).
# Set to 1024: CPU FP64 matmul is O(N^2) and noticeably slow at N > 1024,
# while Stockham already achieves ~1e-4 rel error with far less work.
_DOUBLE_GEMM_THRESHOLD = 1024

# Debug / benchmark toggle — when True, trnfft.fft on the NKI path
# forces Stockham radix-4 dispatch at any power-of-4 N, ignoring the
# usual DFT-GEMM threshold. Set around bench timing blocks only; the
# assertion inside `_fft_via_stockham_nki` is the right failure mode if
# someone flips this on and calls with a non-power-of-4 N. Follows the
# same pattern as `_DFT_GEMM_THRESHOLD` toggling in TestFFT1DSmallN.
_FORCE_STOCKHAM = False

# Same bench toggle for the radix-8 path (Thread B).
_FORCE_STOCKHAM_R8 = False

# Same bench toggle for the mixed-radix path (v0.16, N=1024/2048).
_FORCE_STOCKHAM_MIXED = False

# Bench toggle for the BF16 DFT-GEMM path (v0.17).
_FORCE_BF16_GEMM = False

# Bench toggle for the Ozaki-scheme path (v0.18).
_FORCE_OZAKI = False


def _fft_via_gemm(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Compute FFT as a single complex matmul: X = W @ x (FP32, NKI path).

    Routes onto the Trainium Tensor engine via `complex_gemm`. PSUM
    accumulation on Trainium is always FP32; this path achieves ~1e-3
    relative error at N=256 (the dispatch threshold). For FP64 accuracy,
    use ``precision="double"`` which routes to :func:`_fft_via_gemm_double`.

    Shape handling: flattens leading batch dims to 2D (B, N), does
    ``x @ W`` (W is the N×N DFT matrix), restores original shape.
    """
    from .nki.dispatch import complex_gemm

    n = x.shape[-1]
    orig_shape = x.real.shape
    x_re = x.real.reshape(-1, n).contiguous()
    x_im = x.imag.reshape(-1, n).contiguous()

    sign = 1.0 if inverse else -1.0
    k = torch.arange(n, dtype=x_re.dtype)
    kj = k.unsqueeze(1) * k.unsqueeze(0)
    angles = sign * 2.0 * math.pi * kj / n
    W = ComplexTensor(torch.cos(angles), torch.sin(angles))

    X_2d = complex_gemm(ComplexTensor(x_re, x_im), W)
    X_re = X_2d.real.reshape(orig_shape)
    X_im = X_2d.imag.reshape(orig_shape)
    result = ComplexTensor(X_re, X_im)
    if inverse:
        result = result * (1.0 / n)
    return result


def _fft_via_gemm_double(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """FP64 DFT-GEMM: W @ x computed on CPU in float64.

    Called when ``precision="double"`` and ``n <= _DOUBLE_GEMM_THRESHOLD``.
    Achieves ~1e-14 relative error vs numpy reference.

    Why CPU and not NKI: Trainium's PSUM accumulator is always FP32; even
    FP64 inputs are cast to FP32 before accumulation in ``_complex_gemm_kernel``.
    Explicit CPU computation is the only way to guarantee FP64 precision.
    The result is transferred back to the original device after computation.

    Performance: slower than NKI Stockham (~1e-4 FP32) on hardware. Acceptable
    because ``precision="double"`` is explicitly the accuracy-first path.
    """
    from .complex import complex_matmul

    n = x.shape[-1]
    orig_dtype = x.real.dtype
    orig_device = x.real.device

    x_re_cpu = x.real.detach().cpu().double()
    x_im_cpu = x.imag.detach().cpu().double()

    sign = 1.0 if inverse else -1.0
    k = torch.arange(n, dtype=torch.float64)
    kj = k.unsqueeze(1) * k.unsqueeze(0)
    angles = sign * 2.0 * math.pi * kj / n
    W = ComplexTensor(torch.cos(angles), torch.sin(angles))

    orig_shape = x_re_cpu.shape
    x_re_2d = x_re_cpu.reshape(-1, n).contiguous()
    x_im_2d = x_im_cpu.reshape(-1, n).contiguous()

    X_2d = complex_matmul(ComplexTensor(x_re_2d, x_im_2d), W)
    result = ComplexTensor(
        X_2d.real.reshape(orig_shape),
        X_2d.imag.reshape(orig_shape),
    )
    if inverse:
        result = result * (1.0 / n)

    return ComplexTensor(
        result.real.to(dtype=orig_dtype, device=orig_device),
        result.imag.to(dtype=orig_dtype, device=orig_device),
    )


def _fft_via_gemm_bf16(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """BF16 DFT-GEMM: W and x in BF16, FP32 PSUM accumulation, FP32 output.

    Architectural pattern: nc_matmul with BF16 inputs accumulates to FP32 PSUM
    (hardware invariant on trn1). The BF16 kernel keeps the PSUM as FP32 output
    instead of rounding back to BF16 — "PSUM is a free FP32 accumulator."

    BF16 Tensor Engine throughput is ≈2× FP32. The BF16 W matrix quantisation
    limits accuracy to ≈1e-3 rel error at N=256. For near-FP32 accuracy, use
    :func:`_fft_iterative_refinement` (one correction step at no extra NEFF cost).

    On CPU (no NKI): falls back to ``complex_matmul`` with BF16 inputs — same
    dtype semantics, no throughput benefit but identical correctness test path.
    """
    from .nki.dispatch import complex_gemm_bf16

    n = x.shape[-1]
    orig_shape = x.real.shape

    # Cast x to BF16 for compute. W is built in FP32 then quantized to BF16:
    # computing angles in BF16 introduces large errors for high k*j products
    # (BF16 loses precision for integers > 128). Computing in FP32 and quantizing
    # W entries to BF16 gives the correct ~1e-3 rel error regime.
    x_re = x.real.reshape(-1, n).to(torch.bfloat16).contiguous()
    x_im = x.imag.reshape(-1, n).to(torch.bfloat16).contiguous()

    sign = 1.0 if inverse else -1.0
    k = torch.arange(n, dtype=torch.float32)  # FP32 angles for accuracy
    kj = k.unsqueeze(1) * k.unsqueeze(0)
    angles = sign * 2.0 * math.pi * kj / n
    # Quantize W to BF16 — this is the BF16 DFT-matrix approximation.
    W = ComplexTensor(torch.cos(angles).bfloat16(), torch.sin(angles).bfloat16())

    # complex_gemm_bf16 returns FP32 (PSUM not rounded to BF16).
    X_2d = complex_gemm_bf16(ComplexTensor(x_re, x_im), W)
    X_re = X_2d.real.reshape(orig_shape)
    X_im = X_2d.imag.reshape(orig_shape)
    result = ComplexTensor(X_re, X_im)
    if inverse:
        result = result * (1.0 / n)
    return result


def _fft_iterative_refinement(x: ComplexTensor, inverse: bool, steps: int = 1) -> ComplexTensor:
    """BF16 DFT-GEMM + IR correction steps. Near-FP32 accuracy.

    Algorithm (forward FFT case, inverse analogous):

    .. code-block:: text

        X̂ = fft_bf16(x)                  # BF16 compute, FP32 PSUM output
        for _ in range(steps):
            x_rec = ifft_bf16(X̂)          # reconstruct time domain
            r = x_fp32 - x_rec            # FP32 residual (small after one step)
            δ = fft_bf16(r)               # BF16 FFT of residual
            X̂ = X̂ + δ                    # apply correction

    Cost: (1 + steps) BF16 FFTs + steps inverse FFTs.
    One correction step (IR-1) corrects the BF16 W quantisation error, driving
    accuracy to near-FP32. Subsequent steps give diminishing returns.
    """
    # Step 1: BF16 FFT in the requested direction → FP32 output from PSUM.
    x_hat = _fft_via_gemm_bf16(x, inverse)

    for _ in range(steps):
        # Reconstruct the input from the current estimate.
        # Apply the inverse transform of x_hat (opposite direction + scaling).
        # _fft_via_gemm_bf16 with inverse=True already applies the 1/N factor.
        x_rec = _fft_via_gemm_bf16(x_hat, not inverse)

        # Residual: original input minus reconstruction (in FP32 for accuracy).
        r = ComplexTensor(
            x.real.float() - x_rec.real,
            x.imag.float() - x_rec.imag,
        )

        # BF16 FFT of the residual, same direction as the original transform.
        delta = _fft_via_gemm_bf16(r, inverse)
        x_hat = ComplexTensor(x_hat.real + delta.real, x_hat.imag + delta.imag)

    return x_hat


def _ozaki_split_bf16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split FP32 tensor into BF16 high + low parts (matches dispatch._ozaki_split_bf16)."""
    x_high = x.bfloat16()
    x_low = (x - x_high.float()).bfloat16()
    return x_high, x_low


def _fft_via_ozaki(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Ozaki-scheme DFT-GEMM: 1-level BF16 split, 3 matmuls, FP32 accumulation.

    Decomposes W and x into BF16 high/low parts, calls _fft_via_gemm_bf16 once
    per cross-term (3 total), and sums the FP32 results. Uses _fft_via_gemm_bf16
    directly so each term takes exactly the same hardware path as the working
    BF16 benchmark.

    Expected accuracy: O(sqrt(N) * u_bf16^2) ≈ 8e-6–1.6e-5 rel error at N=64–256.
    Cost: 3× BF16 DFT-GEMM ≈ 2× FP32 DFT-GEMM.
    """
    n = x.shape[-1]
    orig_shape = x.real.shape

    sign = 1.0 if inverse else -1.0
    k = torch.arange(n, dtype=torch.float32)
    kj = k.unsqueeze(1) * k.unsqueeze(0)
    angles = sign * 2.0 * math.pi * kj / n
    W_r_fp32 = torch.cos(angles)  # (n, n) FP32
    W_i_fp32 = torch.sin(angles)

    # Ozaki splits (CPU → BF16)
    x_re_fp32 = x.real.reshape(-1, n).contiguous()
    x_im_fp32 = x.imag.reshape(-1, n).contiguous()
    x_r_h, x_r_l = _ozaki_split_bf16(x_re_fp32)
    x_i_h, x_i_l = _ozaki_split_bf16(x_im_fp32)
    W_r_h, W_r_l = _ozaki_split_bf16(W_r_fp32)
    W_i_h, W_i_l = _ozaki_split_bf16(W_i_fp32)

    def _term(xr, xi, wr, wi) -> ComplexTensor:
        """One BF16 DFT-GEMM term using the validated BF16 kernel path."""
        # Build a fake ComplexTensor carrying BF16 split parts, then call
        # _fft_via_gemm_bf16's kernel directly.  We replicate just the matmul
        # portion (no angle recomputation) via complex_gemm_bf16.
        from .nki.dispatch import complex_gemm_bf16

        return complex_gemm_bf16(ComplexTensor(xr, xi), ComplexTensor(wr, wi))

    # Three cross-terms: hh + hl + lh  (ll is O(u_bf16^4), omitted)
    hh = _term(x_r_h, x_i_h, W_r_h, W_i_h)
    hl = _term(x_r_h, x_i_h, W_r_l, W_i_l)
    lh = _term(x_r_l, x_i_l, W_r_h, W_i_h)

    X_re = (hh.real + hl.real + lh.real).reshape(orig_shape)
    X_im = (hh.imag + hl.imag + lh.imag).reshape(orig_shape)
    result = ComplexTensor(X_re, X_im)
    if inverse:
        result = result * (1.0 / n)
    return result


def _cooley_tukey_nki(x: ComplexTensor, inverse: bool, precision: str = "fast") -> ComplexTensor:
    """Autograd-aware entry point for the NKI FFT path.

    Routes through ``_FFTFn`` (``torch.autograd.Function``) so gradients can
    flow through ``loss.backward()``. The raw forward-only path is
    :func:`_cooley_tukey_nki_nograd`; the autograd Function uses the analytic
    adjoint (IFFT*n for FFT, FFT/n for IFFT) rather than auto-diffing the
    butterfly, which is cheap since FFT is linear.

    ``precision`` selects between the stock butterfly kernel and the Kahan
    (compensated twoProd) variant. Only "fast" and "kahan" meaningfully differ
    on the NKI path; "double" doesn't reach here because NKI is FP32-only.
    """
    from .nki.autograd import fft_autograd

    y_real, y_imag = fft_autograd(x.real, x.imag, inverse, precision)
    return ComplexTensor(y_real, y_imag)


def _is_power_of_four(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0 and (int(math.log2(n)) & 1) == 0


def _stockham_perm_indices(
    log4n: int, B_pad: int, n: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Precompute pack/unpack flat permutation indices for all Stockham stages.

    Returns ``[(pack_idx_s, unpack_idx_s), ...]`` (CPU int64), one pair per stage.

    These replace the reshape+permute+contiguous+reshape chain in the stage loop:

    .. code-block:: python

        # Pack (B_pad*n → total_groups*4):
        re_groups = re.view(-1)[pack_idx].reshape(total_groups, 4)
        # Unpack (total_groups*4 → B_pad*n):
        re = out_re.view(-1)[unpack_idx].reshape(B_pad, n)

    Index derivation
    ----------------
    Pack: ``re_groups[g, k] = re.view(-1)[b*n + l*(4M) + k*M + m]``
    where ``b = g//(L*M)``, ``l = (g%(L*M))//M``, ``m = g%M``.

    Unpack: ``re.view(-1)[b*n + j] = out_re.view(-1)[b*n + g_j*4 + k_j]``
    where ``k_j = j//(L*M)``, ``l_j = (j%(L*M))//M``, ``m_j = j%M``,
    ``g_j = l_j*M + m_j``.  The batch offset in ``out_re.view(-1)`` is also
    ``b*n`` because ``total_groups * 4 = B_pad * n``.
    """
    total_groups = B_pad * n // 4
    result: list[tuple[torch.Tensor, torch.Tensor]] = []

    for s in range(log4n):
        L = 1 << (2 * s)
        M = n // (4 * L)

        # Pack indices — shape (total_groups * 4,)
        g = torch.arange(total_groups, dtype=torch.int64)
        b_g = g // (L * M)
        l_g = (g % (L * M)) // M
        m_g = g % M
        k4 = torch.arange(4, dtype=torch.int64)
        # flat_pack[g, k] = b*n + l*(4M) + k*M + m
        flat_pack = (b_g * n + l_g * (4 * M) + m_g).unsqueeze(1) + k4.unsqueeze(0) * M
        pack_idx = flat_pack.reshape(-1)

        # Unpack indices — shape (B_pad * n,)
        j = torch.arange(n, dtype=torch.int64)
        k_j = j // (L * M)
        l_j = (j % (L * M)) // M
        m_j = j % M
        g_j = l_j * M + m_j  # group index within one batch
        per_b = g_j * 4 + k_j  # flat index within one batch's slice of out_re
        b_all = torch.arange(B_pad, dtype=torch.int64)
        flat_unpack = b_all.unsqueeze(1) * n + per_b.unsqueeze(0)  # (B_pad, n)
        unpack_idx = flat_unpack.reshape(-1)

        result.append((pack_idx, unpack_idx))

    return result


def _fft_via_stockham_nki(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Radix-4 Stockham FFT driver — dispatches log_4(N) NKI stages.

    Preconditions: ``x.shape[-1]`` is a power of 4. All per-stage twiddle
    factors are precomputed on CPU and transferred to the XLA device before
    the stage loop. This eliminates the per-stage H→D transfer cost that was
    the primary driver overhead (profiling: 2026-04-16).

    After log_4(N) stages the output is in natural index order (Stockham
    property — no bit-reversal permutation required).

    Inverse via the conjugate trick: ``ifft(X) = conj(fft(conj(X))) / N``.

    Twiddle precomputation: ``total_groups = B_pad * L * M = B_pad * N/4`` is
    constant across stages (L = 4^s, M = N/(4·4^s), product = N/4). All
    log_4(N) twiddle tensors have the same shape ``(total_groups, 4)``.

    Transfer strategy: separate tensors per stage (not slices of a stacked
    tensor). XLA/Neuron slice ops on device tensors force new HLO programs and
    bust the NEFF cache; standalone per-stage tensors reuse the cached NEFF.
    """
    from .nki.dispatch import _use_simulator
    from .nki.stockham import stockham_radix4_stage_kernel

    n = x.shape[-1]
    assert _is_power_of_four(n), f"Stockham radix-4 requires N=4^k; got N={n}"

    use_sim = _use_simulator()
    if not use_sim:
        import torch_xla

    orig_shape = x.real.shape
    flat_re = x.real.reshape(-1, n).contiguous()
    flat_im = x.imag.reshape(-1, n).contiguous()

    # Conjugate trick for inverse.
    if inverse:
        flat_im = -flat_im

    B = flat_re.shape[0]
    # Partition-dim tile alignment. For non-power-of-2 B, pad to the next
    # multiple of PMAX so groups_chunk divides total_groups cleanly.
    PMAX = 128

    def _is_pow2(v: int) -> bool:
        return v > 0 and (v & (v - 1)) == 0

    if _is_pow2(B):
        B_pad = B
        pad_re = flat_re
        pad_im = flat_im
    else:
        B_pad = ((B + PMAX - 1) // PMAX) * PMAX
        padding = torch.zeros(B_pad - B, n, dtype=flat_re.dtype)
        pad_re = torch.cat([flat_re, padding], dim=0)
        pad_im = torch.cat([flat_im, padding], dim=0)

    orig_device = x.real.device
    device = None if use_sim else torch_xla.device()

    log4n = int(math.log2(n)) // 2
    # total_groups = B_pad * L * M = B_pad * N/4 — constant across all stages.
    total_groups = B_pad * n // 4

    # Precompute ALL twiddle factors on CPU before the stage loop, then
    # transfer to device as a list of standalone tensors — not as slices of
    # a stacked tensor. Slicing an XLA device tensor creates DynamicSlice HLO
    # ops that change the compiled graph and bust the NEFF cache.
    twiddles_cpu = []
    for s in range(log4n):
        L = 1 << (2 * s)
        M = n // (4 * L)
        l_idx = torch.arange(L, dtype=x.real.dtype).view(1, L, 1, 1)
        k_idx = torch.arange(4, dtype=x.real.dtype).view(1, 1, 1, 4)
        ang = -2.0 * math.pi * l_idx * k_idx / (4.0 * L)
        tw_r = torch.cos(ang).expand(B_pad, L, M, 4).contiguous().reshape(total_groups, 4)
        tw_i = torch.sin(ang).expand(B_pad, L, M, 4).contiguous().reshape(total_groups, 4)
        twiddles_cpu.append((tw_r, tw_i))

    re = pad_re if use_sim else pad_re.to(device)
    im = pad_im if use_sim else pad_im.to(device)

    if not use_sim:
        # Transfer all per-stage twiddles to XLA device before the kernel loop.
        # Each entry is an independent tensor; the kernel call sees the same
        # (total_groups, 4) shape at every stage — NEFF cache hit guaranteed.
        twiddles = [(tw_r.to(device), tw_i.to(device)) for tw_r, tw_i in twiddles_cpu]
    else:
        twiddles = twiddles_cpu

    for s in range(log4n):
        L = 1 << (2 * s)
        M = n // (4 * L)
        tw_r, tw_i = twiddles[s]

        # Pack stage-s input into (total_groups, 4): each partition row is
        # one 4-point DFT group with k as the contiguous inner axis.
        # Transpose HLO (permute+contiguous) is faster than GatherOp on Neuron —
        # gather was measured 11–39% slower in the v0.14 Thread C experiment.
        re_4d = re.reshape(B_pad, L, 4, M)
        im_4d = im.reshape(B_pad, L, 4, M)
        re_groups = re_4d.permute(0, 1, 3, 2).contiguous().reshape(total_groups, 4)
        im_groups = im_4d.permute(0, 1, 3, 2).contiguous().reshape(total_groups, 4)

        if use_sim:
            import nki as _nki

            out_re_np, out_im_np = _nki.simulate(stockham_radix4_stage_kernel)(
                re_groups.detach().cpu().numpy(),
                im_groups.detach().cpu().numpy(),
                tw_r.detach().cpu().numpy(),
                tw_i.detach().cpu().numpy(),
            )
            out_re = torch.from_numpy(np.asarray(out_re_np))
            out_im = torch.from_numpy(np.asarray(out_im_np))
        else:
            out_re, out_im = stockham_radix4_stage_kernel(re_groups, im_groups, tw_r, tw_i)

        # Stockham output permutation: (B, L, M, 4) -> (B, 4, L, M) -> (B, N)
        out_re_4d = out_re.reshape(B_pad, L, M, 4)
        out_im_4d = out_im.reshape(B_pad, L, M, 4)
        re = out_re_4d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)
        im = out_im_4d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)

    if not use_sim:
        re = re.to(orig_device)
        im = im.to(orig_device)
    re = re[:B].reshape(orig_shape)
    im = im[:B].reshape(orig_shape)

    if inverse:
        im = -im
        re = re / n
        im = im / n
    return ComplexTensor(re, im)


def _is_power_of_eight(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0 and int(math.log2(n)) % 3 == 0


def _mixed_radix_plan(n: int) -> list[int]:
    """Optimal [8^a, 4^b] stage sequence for power-of-2 n (same as stockham.py)."""
    k = int(math.log2(n))
    a = k // 3
    while (k - 3 * a) % 2 != 0:
        a -= 1
    return [8] * a + [4] * ((k - 3 * a) // 2)


def _fft_via_stockham_nki_r8(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Radix-8 Stockham FFT driver — dispatches log_8(N) NKI stages.

    Preconditions: ``x.shape[-1]`` is a power of 8 (8, 64, 512, 4096, ...).
    The W_8 DFT matrix is precomputed on CPU and passed to each stage kernel
    as a shared ``(8, 8)`` complex tensor.  Twiddle factors follow the same
    precomputation pattern as the radix-4 driver.

    Per-stage kernel structure: Vector-engine twiddle multiply, then
    Tensor-engine W_8 matmul via ``nc_matmul``.  An HBM scratch buffer
    bridges the two phases (``nl.load_transpose2d`` requires HBM source).

    Coverage: N=512 (3 stages vs 9 butterfly) and N=4096 (4 stages vs 6
    radix-4).  Hardware bench will determine if the Tensor-engine W_8 win
    and lower stage count outweigh the scratch round-trip cost.
    """
    from .nki.dispatch import _use_simulator
    from .nki.stockham import stockham_radix8_w8_kernel

    n = x.shape[-1]
    assert _is_power_of_eight(n), f"Radix-8 requires N=8^k; got N={n}"

    use_sim = _use_simulator()
    if not use_sim:
        import torch_xla

    orig_shape = x.real.shape
    flat_re = x.real.reshape(-1, n).contiguous()
    flat_im = x.imag.reshape(-1, n).contiguous()

    if inverse:
        flat_im = -flat_im

    B = flat_re.shape[0]
    PMAX = 128

    def _is_pow2(v: int) -> bool:
        return v > 0 and (v & (v - 1)) == 0

    if _is_pow2(B):
        B_pad = B
        pad_re = flat_re
        pad_im = flat_im
    else:
        B_pad = ((B + PMAX - 1) // PMAX) * PMAX
        padding = torch.zeros(B_pad - B, n, dtype=flat_re.dtype)
        pad_re = torch.cat([flat_re, padding], dim=0)
        pad_im = torch.cat([flat_im, padding], dim=0)

    orig_device = x.real.device
    device = None if use_sim else torch_xla.device()

    log8n = int(math.log2(n)) // 3
    total_groups = B_pad * n // 8

    # Precompute W_8 on CPU (shared across all stages, all groups).
    # W_8[k, n_] = exp(-2πi·k·n_/8); symmetric so W_8^T = W_8.
    k_idx = torch.arange(8, dtype=x.real.dtype)
    n_idx = torch.arange(8, dtype=x.real.dtype)
    ang_w8 = -2.0 * math.pi * k_idx.unsqueeze(1) * n_idx.unsqueeze(0) / 8
    w8_re_cpu = torch.cos(ang_w8)  # (8, 8)
    w8_im_cpu = torch.sin(ang_w8)  # (8, 8)

    # Precompute ALL twiddle factors on CPU before the stage loop.
    # Same NEFF-cache-safe pattern as radix-4: standalone tensors, not slices.
    twiddles_cpu = []
    for s in range(log8n):
        L = 1 << (3 * s)  # 8^s
        M = n // (8 * L)
        l_idx = torch.arange(L, dtype=x.real.dtype).view(1, L, 1, 1)
        k8_idx = torch.arange(8, dtype=x.real.dtype).view(1, 1, 1, 8)
        ang = -2.0 * math.pi * l_idx * k8_idx / (8.0 * L)
        tw_r = torch.cos(ang).expand(B_pad, L, M, 8).contiguous().reshape(total_groups, 8)
        tw_i = torch.sin(ang).expand(B_pad, L, M, 8).contiguous().reshape(total_groups, 8)
        twiddles_cpu.append((tw_r, tw_i))

    re = pad_re if use_sim else pad_re.to(device)
    im = pad_im if use_sim else pad_im.to(device)

    if not use_sim:
        twiddles = [(tw_r.to(device), tw_i.to(device)) for tw_r, tw_i in twiddles_cpu]
        w8_re = w8_re_cpu.to(device)
        w8_im = w8_im_cpu.to(device)
    else:
        twiddles = twiddles_cpu
        w8_re = w8_re_cpu
        w8_im = w8_im_cpu

    for s in range(log8n):
        L = 1 << (3 * s)
        M = n // (8 * L)
        tw_r, tw_i = twiddles[s]

        # Pack: (B_pad, N) → (B_pad, L, 8, M) → permute(0,1,3,2) → (B_pad, L, M, 8)
        # → reshape (total_groups, 8)
        re_8d = re.reshape(B_pad, L, 8, M)
        im_8d = im.reshape(B_pad, L, 8, M)
        re_groups = re_8d.permute(0, 1, 3, 2).contiguous().reshape(total_groups, 8)
        im_groups = im_8d.permute(0, 1, 3, 2).contiguous().reshape(total_groups, 8)

        # Twiddle multiply (XLA element-wise, same as radix-4 driver).
        # nl.load_transpose2d requires external HBM args; twiddle stays in PyTorch.
        pre_re = re_groups * tw_r - im_groups * tw_i
        pre_im = re_groups * tw_i + im_groups * tw_r

        if use_sim:
            import nki as _nki

            out_re_np, out_im_np = _nki.simulate(stockham_radix8_w8_kernel)(
                pre_re.detach().cpu().numpy(),
                pre_im.detach().cpu().numpy(),
                w8_re.detach().cpu().numpy(),
                w8_im.detach().cpu().numpy(),
            )
            out_re = torch.from_numpy(np.asarray(out_re_np))
            out_im = torch.from_numpy(np.asarray(out_im_np))
        else:
            out_re, out_im = stockham_radix8_w8_kernel(pre_re, pre_im, w8_re, w8_im)

        # Stockham output permutation: (B, L, M, 8) → (B, 8, L, M) → (B, N)
        out_re_8d = out_re.reshape(B_pad, L, M, 8)
        out_im_8d = out_im.reshape(B_pad, L, M, 8)
        re = out_re_8d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)
        im = out_im_8d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)

    if not use_sim:
        re = re.to(orig_device)
        im = im.to(orig_device)
    re = re[:B].reshape(orig_shape)
    im = im[:B].reshape(orig_shape)

    if inverse:
        im = -im
        re = re / n
        im = im / n
    return ComplexTensor(re, im)


def _fft_via_stockham_nki_mixed(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Mixed-radix [8^a, 4^b] Stockham FFT driver.

    Computes the optimal stage decomposition for any power-of-2 N, then
    executes the stages by alternating between the radix-8 Tensor-engine
    kernel and the radix-4 Vector-engine kernel as dictated by the plan.

    Coverage that improves over existing paths:
      N=1024: [8,8,4,4] = 4 stages  (vs radix-4's 5)
      N=2048: [8,8,8,4] = 4 stages  (vs butterfly's 11 — new Stockham coverage)

    All per-stage twiddles are precomputed on CPU and transferred once before
    the loop (same NEFF-cache-safe pattern as the pure radix drivers).
    W₈ is precomputed once and shared across all radix-8 stages.
    """
    from .nki.dispatch import _use_simulator
    from .nki.stockham import stockham_radix4_stage_kernel, stockham_radix8_w8_kernel

    n = x.shape[-1]
    plan = _mixed_radix_plan(n)

    use_sim = _use_simulator()
    if not use_sim:
        import torch_xla

    orig_shape = x.real.shape
    flat_re = x.real.reshape(-1, n).contiguous()
    flat_im = x.imag.reshape(-1, n).contiguous()

    if inverse:
        flat_im = -flat_im

    B = flat_re.shape[0]
    PMAX = 128

    def _is_pow2(v: int) -> bool:
        return v > 0 and (v & (v - 1)) == 0

    if _is_pow2(B):
        B_pad, pad_re, pad_im = B, flat_re, flat_im
    else:
        B_pad = ((B + PMAX - 1) // PMAX) * PMAX
        padding = torch.zeros(B_pad - B, n, dtype=flat_re.dtype)
        pad_re = torch.cat([flat_re, padding], dim=0)
        pad_im = torch.cat([flat_im, padding], dim=0)

    orig_device = x.real.device
    device = None if use_sim else torch_xla.device()

    # Precompute W₈ (shared for all r=8 stages, if any).
    has_r8 = any(r == 8 for r in plan)
    if has_r8:
        k8 = torch.arange(8, dtype=x.real.dtype)
        ang_w8 = -2.0 * math.pi * k8.unsqueeze(1) * k8.unsqueeze(0) / 8
        w8_re_cpu = torch.cos(ang_w8)
        w8_im_cpu = torch.sin(ang_w8)
    else:
        w8_re_cpu = w8_im_cpu = None

    # Precompute per-stage twiddles on CPU. L evolves with each stage.
    twiddles_cpu = []
    L_pre = 1
    for r in plan:
        M = n // (r * L_pre)
        total_groups_s = B_pad * L_pre * M
        l_idx = torch.arange(L_pre, dtype=x.real.dtype).view(1, L_pre, 1, 1)
        k_idx = torch.arange(r, dtype=x.real.dtype).view(1, 1, 1, r)
        ang = -2.0 * math.pi * l_idx * k_idx / (float(r) * L_pre)
        tw_r = torch.cos(ang).expand(B_pad, L_pre, M, r).contiguous().reshape(total_groups_s, r)
        tw_i = torch.sin(ang).expand(B_pad, L_pre, M, r).contiguous().reshape(total_groups_s, r)
        twiddles_cpu.append((tw_r, tw_i))
        L_pre *= r

    re = pad_re if use_sim else pad_re.to(device)
    im = pad_im if use_sim else pad_im.to(device)

    if not use_sim:
        twiddles = [(tw_r.to(device), tw_i.to(device)) for tw_r, tw_i in twiddles_cpu]
        w8_re = w8_re_cpu.to(device) if has_r8 else None
        w8_im = w8_im_cpu.to(device) if has_r8 else None
    else:
        twiddles = twiddles_cpu
        w8_re, w8_im = w8_re_cpu, w8_im_cpu

    L = 1
    for s, r in enumerate(plan):
        M = n // (r * L)
        tw_r, tw_i = twiddles[s]
        total_groups_s = B_pad * L * M

        re_rd = re.reshape(B_pad, L, r, M)
        im_rd = im.reshape(B_pad, L, r, M)
        re_groups = re_rd.permute(0, 1, 3, 2).contiguous().reshape(total_groups_s, r)
        im_groups = im_rd.permute(0, 1, 3, 2).contiguous().reshape(total_groups_s, r)

        if r == 8:
            pre_re = re_groups * tw_r - im_groups * tw_i
            pre_im = re_groups * tw_i + im_groups * tw_r
            if use_sim:
                import nki as _nki

                out_re_np, out_im_np = _nki.simulate(stockham_radix8_w8_kernel)(
                    pre_re.detach().cpu().numpy(),
                    pre_im.detach().cpu().numpy(),
                    w8_re.detach().cpu().numpy(),
                    w8_im.detach().cpu().numpy(),
                )
                out_re = torch.from_numpy(np.asarray(out_re_np))
                out_im = torch.from_numpy(np.asarray(out_im_np))
            else:
                out_re, out_im = stockham_radix8_w8_kernel(pre_re, pre_im, w8_re, w8_im)
        else:
            if use_sim:
                import nki as _nki

                out_re_np, out_im_np = _nki.simulate(stockham_radix4_stage_kernel)(
                    re_groups.detach().cpu().numpy(),
                    im_groups.detach().cpu().numpy(),
                    tw_r.detach().cpu().numpy(),
                    tw_i.detach().cpu().numpy(),
                )
                out_re = torch.from_numpy(np.asarray(out_re_np))
                out_im = torch.from_numpy(np.asarray(out_im_np))
            else:
                out_re, out_im = stockham_radix4_stage_kernel(re_groups, im_groups, tw_r, tw_i)

        out_re_rd = out_re.reshape(B_pad, L, M, r)
        out_im_rd = out_im.reshape(B_pad, L, M, r)
        re = out_re_rd.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)
        im = out_im_rd.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)
        L *= r

    if not use_sim:
        re = re.to(orig_device)
        im = im.to(orig_device)
    re = re[:B].reshape(orig_shape)
    im = im[:B].reshape(orig_shape)

    if inverse:
        im = -im
        re = re / n
        im = im / n
    return ComplexTensor(re, im)


def _cooley_tukey_nki_nograd(
    x: ComplexTensor, inverse: bool, precision: str = "fast"
) -> ComplexTensor:
    """Cooley-Tukey FFT using batched NKI butterfly kernel on Trainium.

    Accepts any shape; leading dims are flattened into a single batch dim B
    and passed to the kernel as (B, n). The kernel vectorizes across B in a
    single call per stage — no Python loop over batch rows.

    ``precision`` picks the butterfly variant: "fast" (default, stock kernel)
    or "kahan" (compensated twoProd complex multiply — ~2× slower).

    This is the forward-only path (no autograd). Call :func:`_cooley_tukey_nki`
    instead if autograd support is needed.

    For N <= _DFT_GEMM_THRESHOLD the butterfly path is bypassed in favor of
    :func:`_fft_via_gemm`, which routes onto the Tensor engine (one matmul)
    instead of the Vector engine (log2(N) butterfly stages). "fast" precision
    only — the kahan variant stays on the butterfly path so the compensated
    complex multiply remains available for users who explicitly want it.
    """
    from .nki.butterfly import butterfly_stage_kernel, butterfly_stage_kernel_kahan
    from .nki.dispatch import _use_simulator

    kernel = butterfly_stage_kernel_kahan if precision == "kahan" else butterfly_stage_kernel
    use_sim = _use_simulator()
    if not use_sim:
        import torch_xla

    n = x.shape[-1]

    # Bench-toggles: force a specific path ahead of all threshold checks.
    if _FORCE_OZAKI:
        return _fft_via_ozaki(x, inverse, levels=2)
    if _FORCE_BF16_GEMM:
        return _fft_via_gemm_bf16(x, inverse)
    if _FORCE_STOCKHAM_MIXED and precision != "kahan":
        return _fft_via_stockham_nki_mixed(x, inverse)
    if _FORCE_STOCKHAM_R8 and precision != "kahan":
        return _fft_via_stockham_nki_r8(x, inverse)
    if _FORCE_STOCKHAM and precision != "kahan":
        return _fft_via_stockham_nki(x, inverse)

    # "double" mode: NKI PSUM is always FP32, so bypass NKI and use CPU FP64.
    if precision == "double" and n <= _DOUBLE_GEMM_THRESHOLD:
        return _fft_via_gemm_double(x, inverse)

    # "bf16" mode: BF16 compute on Tensor Engine, FP32 PSUM → FP32 output.
    if precision == "bf16" and n <= _DFT_GEMM_THRESHOLD:
        return _fft_via_gemm_bf16(x, inverse)

    # "bf16_refined" mode: BF16 + one iterative correction → near-FP32 accuracy.
    if precision == "bf16_refined" and n <= _DFT_GEMM_THRESHOLD:
        return _fft_iterative_refinement(x, inverse, steps=1)

    # "ozaki" mode: 1-level BF16 split, 3 matmuls, O(sqrt(N)*u_bf16^2) error.
    if precision == "ozaki" and n <= _DFT_GEMM_THRESHOLD:
        return _fft_via_ozaki(x, inverse)

    if n <= _DFT_GEMM_THRESHOLD and precision != "kahan":
        return _fft_via_gemm(x, inverse)

    # Mixed-radix Stockham (v0.16): optimal [8^a, 4^b] plan for any power-of-2.
    # Only dispatched when mixed gives fewer stages than the best pure alternative.
    # Key cases: N=1024 (4 vs 5 radix-4 stages) and N=2048 (4 vs 11 butterfly).
    if n > 1 and (n & (n - 1)) == 0 and precision != "kahan":
        _plan = _mixed_radix_plan(n)
        _r8 = int(math.log2(n)) // 3 if _is_power_of_eight(n) else None
        _r4 = int(math.log2(n)) // 2 if _is_power_of_four(n) else None
        _best = min(s for s in [_r8, _r4, int(math.log2(n))] if s is not None)
        if len(_plan) < _best:
            return _fft_via_stockham_nki_mixed(x, inverse)

    # Radix-8 Stockham (Thread B): N=512 (3 stages) and N=4096 (4 stages).
    if _is_power_of_eight(n) and precision != "kahan":
        return _fft_via_stockham_nki_r8(x, inverse)

    # Radix-4 Stockham: hardware-validated, 6–9% faster than butterfly.
    if _is_power_of_four(n) and precision != "kahan":
        return _fft_via_stockham_nki(x, inverse)
    log2n = int(math.log2(n))
    assert 1 << log2n == n, f"Not power of 2: {n}"

    # Normalize to 2D (B, n). Restore original shape at the end.
    orig_shape = x.real.shape
    flat_re = x.real.reshape(-1, n).contiguous()
    flat_im = x.imag.reshape(-1, n).contiguous()
    B_orig = flat_re.shape[0]

    # The NKI kernel's partition tiling uses chunk size PMAX=128 and requires
    # total_groups = B * num_groups to be divisible by PMAX at every stage
    # where total_groups > PMAX. For power-of-2 B and power-of-2 n, every
    # total_groups value is also a power of 2 — so whenever total_groups > 128
    # it is automatically divisible by 128. No padding needed.
    #
    # For non-power-of-2 B (STFT's num_frames=33, for example), pad B up to
    # the next multiple of PMAX so the constraint holds. Unbatched FFT
    # (B=1) is already a power of 2 and takes the zero-copy path.
    PMAX = 128

    def _is_pow2(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0

    if _is_pow2(B_orig):
        B = B_orig
        pad_re = flat_re
        pad_im = flat_im
    else:
        B = ((B_orig + PMAX - 1) // PMAX) * PMAX
        padding = torch.zeros(B - B_orig, n, dtype=flat_re.dtype)
        pad_re = torch.cat([flat_re, padding], dim=0)
        pad_im = torch.cat([flat_im, padding], dim=0)

    sign = 1.0 if inverse else -1.0

    orig_device = x.real.device
    device = None if use_sim else torch_xla.device()

    # Bit-reversal permutation along the last dim.
    indices = _bit_reverse_indices(n, log2n)
    re = pad_re[..., indices]
    im = pad_im[..., indices]
    if not use_sim:
        re = re.to(device)
        im = im.to(device)

    # Run butterfly stages via batched NKI kernel.
    for s in range(log2n):
        m = 1 << (s + 1)
        half = m >> 1
        num_groups = n // m
        total_groups = B * num_groups

        # Precompute twiddle factors for this stage. Expand to (total_groups, half)
        # so every batch row / group row has matching partition-dim values.
        angles = sign * 2.0 * math.pi * torch.arange(half, dtype=x.real.dtype) / m
        tw_re_1d = torch.cos(angles)
        tw_im_1d = torch.sin(angles)
        tw_re_bcast = tw_re_1d.unsqueeze(0).expand(total_groups, half).contiguous()
        tw_im_bcast = tw_im_1d.unsqueeze(0).expand(total_groups, half).contiguous()

        if use_sim:
            import nki as _nki

            re_np, im_np = _nki.simulate(kernel)(
                re.detach().cpu().numpy(),
                im.detach().cpu().numpy(),
                tw_re_bcast.detach().cpu().numpy(),
                tw_im_bcast.detach().cpu().numpy(),
                n,
                s,
            )
            re = torch.from_numpy(np.asarray(re_np))
            im = torch.from_numpy(np.asarray(im_np))
        else:
            tw_re_bcast = tw_re_bcast.to(device)
            tw_im_bcast = tw_im_bcast.to(device)
            re, im = kernel(re, im, tw_re_bcast, tw_im_bcast, n, s)

    if not use_sim:
        re = re.to(orig_device)
        im = im.to(orig_device)
    re = re[:B_orig].reshape(orig_shape)
    im = im[:B_orig].reshape(orig_shape)
    result = ComplexTensor(re, im)
    if inverse:
        result = result * (1.0 / n)
    return result


def _complex_mul_kahan(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """Elementwise complex multiply with 2Prod-compensated accumulation.

    Standard complex multiply is ``c_re = a_re*b_re - a_im*b_im``,
    ``c_im = a_re*b_im + a_im*b_re``. At FP32, catastrophic cancellation
    in the difference (``a_re*b_re ≈ a_im*b_im``) costs mantissa bits.
    We use Kahan/Dekker 2Prod to recover the lost part:

        twoProd(x, y) = (hi, lo)  where  hi + lo = x*y exactly.

    Then ``c_re = (hi_rr - hi_ii) + (lo_rr - lo_ii)`` and similarly for
    ``c_im``. The ``lo`` terms compensate the rounding error.

    Without an FMA primitive, 2Prod is done via Dekker's split:
        split(x) = (hi, lo) where hi+lo = x exactly, with hi = x rounded
                   to half the mantissa.
    """
    ar, ai = a.real, a.imag
    br, bi = b.real, b.imag

    def _two_prod(x, y):
        # Dekker's 2Prod via splitting. For FP32, split at bit 12.
        # split factor = 2^12 + 1 = 4097
        C = 4097.0
        xc = C * x
        xh = xc - (xc - x)
        xl = x - xh
        yc = C * y
        yh = yc - (yc - y)
        yl = y - yh
        hi = x * y
        lo = ((xh * yh - hi) + xh * yl + xl * yh) + xl * yl
        return hi, lo

    hi_rr, lo_rr = _two_prod(ar, br)
    hi_ii, lo_ii = _two_prod(ai, bi)
    hi_ri, lo_ri = _two_prod(ar, bi)
    hi_ir, lo_ir = _two_prod(ai, br)

    c_re = (hi_rr - hi_ii) + (lo_rr - lo_ii)
    c_im = (hi_ri + hi_ir) + (lo_ri + lo_ir)
    return ComplexTensor(c_re, c_im)


def _bluestein(
    x: ComplexTensor,
    inverse: bool,
    padded_n: int | None = None,
    precision: str = "fast",
) -> ComplexTensor:
    """Bluestein's algorithm (chirp-z) for arbitrary-size FFT.

    Converts length-N DFT into circular convolution of length M >= 2N-1
    (M is next power of 2), computed via three power-of-2 FFTs.

    ``precision`` modes:
      - "fast":   straight FP32 (or input dtype), chain accumulates ~2e-2
                  relative error at N >= 500.
      - "kahan":  compensated complex multiply at the two chirp multiplies
                  and at the Y*H product. Reduces error ~10-100×.
      - "double": promote entire host-side math to FP64, then cast back.
                  Largest precision win (~6+ orders of magnitude) but
                  Bluestein-only — power-of-2 FFTs are unaffected, and NKI
                  kernels stay FP32 throughout the rest of the library.
    """
    # Dtype promotion for "double" mode. Cast back at the end.
    orig_dtype = x.real.dtype
    if precision == "double":
        x = ComplexTensor(x.real.double(), x.imag.double())

    n = x.shape[-1]
    m = padded_n if padded_n is not None else (1 << (2 * n - 2).bit_length())

    sign = 1.0 if inverse else -1.0

    # Chirp sequence: W_N^(k^2/2) = exp(sign * πi * k^2 / N)
    k = torch.arange(n, dtype=x.dtype)
    chirp_angles = sign * math.pi * k * k / n
    chirp_re = torch.cos(chirp_angles)
    chirp_im = torch.sin(chirp_angles)
    chirp = ComplexTensor(chirp_re, chirp_im)

    # Step 1: y[n] = x[n] * chirp[n]
    if precision == "kahan":
        y = _complex_mul_kahan(x, chirp)
    else:
        y = x * chirp

    # Step 2: Zero-pad y to length m
    batch_shape = x.shape[:-1]
    y_pad_re = torch.zeros(*batch_shape, m, dtype=x.dtype)
    y_pad_im = torch.zeros(*batch_shape, m, dtype=x.dtype)
    y_pad_re[..., :n] = y.real
    y_pad_im[..., :n] = y.imag
    y_padded = ComplexTensor(y_pad_re, y_pad_im)

    # Step 3: Build filter h = conj(chirp) with circular wrap
    h_re = torch.zeros(m, dtype=x.dtype)
    h_im = torch.zeros(m, dtype=x.dtype)
    h_re[:n] = chirp_re
    h_im[:n] = -chirp_im  # conjugate
    for i in range(1, n):
        h_re[m - i] = chirp_re[i]
        h_im[m - i] = -chirp_im[i]  # conjugate
    h = ComplexTensor(h_re, h_im)

    # Step 4: Circular convolution via FFT
    Y = _cooley_tukey(y_padded, inverse=False, precision=precision)
    H = _cooley_tukey(
        h.unsqueeze(0) if len(batch_shape) > 0 else h,
        inverse=False,
        precision=precision,
    )
    if precision == "kahan":
        product = _complex_mul_kahan(Y, H)
    else:
        product = Y * H
    conv = _cooley_tukey(product, inverse=True, precision=precision)

    # Step 5: Multiply by chirp and take first N elements
    result_re = conv.real[..., :n]
    result_im = conv.imag[..., :n]
    result_shifted = ComplexTensor(result_re, result_im)
    if precision == "kahan":
        result = _complex_mul_kahan(result_shifted, chirp)
    else:
        result = result_shifted * chirp

    if inverse:
        result = result * (1.0 / n)

    # Cast back to original dtype if we promoted for "double" mode.
    if precision == "double" and result.real.dtype != orig_dtype:
        result = ComplexTensor(result.real.to(orig_dtype), result.imag.to(orig_dtype))

    return result


def _bit_reverse_indices(n: int, bits: int) -> torch.Tensor:
    """Compute bit-reversal permutation indices."""
    result = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        rev = 0
        val = i
        for _ in range(bits):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        result[i] = rev
    return result
