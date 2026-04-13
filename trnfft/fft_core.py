"""
FFT core algorithms.

Iterative radix-2 Cooley-Tukey (decimation-in-time) for power-of-2 sizes.
Bluestein chirp-z for arbitrary sizes.
All operations on split real/imaginary tensors for Trainium compatibility.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Optional

from .complex import ComplexTensor
from .plan import FFTPlan, FFTAlgorithm, create_plan
from .nki.dispatch import _use_nki, HAS_NKI


def fft_core(
    x: ComplexTensor,
    inverse: bool = False,
    plan: Optional[FFTPlan] = None,
    precision: Optional[str] = None,
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
        m = 1 << (s + 1)       # Butterfly group size
        half = m >> 1           # Half group

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
#   Threshold set conservatively at 128 = one PSUM tile
#   (TILE_K=TILE_M=128 in `_complex_gemm_kernel`). At the threshold, CT
#   would do 7 butterfly stages + bit-reversal; DFT-GEMM does one matmul.
#   Empirical threshold refinement is the v0.12 milestone-1 output.
_DFT_GEMM_THRESHOLD = 128


def _fft_via_gemm(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Compute FFT as a single complex matmul: X = W @ x.

    Routes onto the Trainium Tensor engine via `complex_gemm`. Works on CPU
    too (PyTorch matmul fallback) but asymptotically worse than Cooley-Tukey
    there; this path only gets dispatched when `_use_nki()`.

    Shape handling: flattens leading batch dims to 2D (B, N), does
    `x @ W` (W is the N×N symmetric DFT matrix), restores original shape.
    """
    from .nki.dispatch import complex_gemm

    n = x.shape[-1]
    orig_shape = x.real.shape
    x_re = x.real.reshape(-1, n).contiguous()
    x_im = x.imag.reshape(-1, n).contiguous()

    sign = 1.0 if inverse else -1.0
    k = torch.arange(n, dtype=x_re.dtype)
    kj = k.unsqueeze(1) * k.unsqueeze(0)  # (N, N) outer product: W_angle = sign * 2π * k * j / N
    angles = sign * 2.0 * math.pi * kj / n
    W = ComplexTensor(torch.cos(angles), torch.sin(angles))

    # x is (B, N), W is (N, N) symmetric, so x @ W gives (B, N).
    X_2d = complex_gemm(ComplexTensor(x_re, x_im), W)
    X_re = X_2d.real.reshape(orig_shape)
    X_im = X_2d.imag.reshape(orig_shape)
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


def _cooley_tukey_nki_nograd(x: ComplexTensor, inverse: bool, precision: str = "fast") -> ComplexTensor:
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
    import torch_xla
    kernel = butterfly_stage_kernel_kahan if precision == "kahan" else butterfly_stage_kernel

    n = x.shape[-1]
    if n <= _DFT_GEMM_THRESHOLD and precision != "kahan":
        return _fft_via_gemm(x, inverse)

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

    device = torch_xla.device()
    orig_device = x.real.device

    # Bit-reversal permutation along the last dim.
    indices = _bit_reverse_indices(n, log2n)
    re = pad_re[..., indices].to(device)
    im = pad_im[..., indices].to(device)

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
        tw_re_bcast = tw_re_1d.unsqueeze(0).expand(total_groups, half).contiguous().to(device)
        tw_im_bcast = tw_im_1d.unsqueeze(0).expand(total_groups, half).contiguous().to(device)

        re, im = kernel(re, im, tw_re_bcast, tw_im_bcast, n, s)

    re = re.to(orig_device)[:B_orig].reshape(orig_shape)
    im = im.to(orig_device)[:B_orig].reshape(orig_shape)
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
    padded_n: Optional[int] = None,
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
