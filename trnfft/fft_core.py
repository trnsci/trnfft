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

# Debug / benchmark toggle — when True, trnfft.fft on the NKI path
# forces Stockham radix-4 dispatch at any power-of-4 N, ignoring the
# usual DFT-GEMM threshold. Set around bench timing blocks only; the
# assertion inside `_fft_via_stockham_nki` is the right failure mode if
# someone flips this on and calls with a non-power-of-4 N. Follows the
# same pattern as `_DFT_GEMM_THRESHOLD` toggling in TestFFT1DSmallN.
_FORCE_STOCKHAM = False


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

    # Precompute ALL twiddle factors and pack/unpack permutation indices on CPU
    # before the stage loop, then transfer to device as standalone tensors.
    # Slicing XLA device tensors creates DynamicSlice HLO ops that bust NEFF cache.
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

    perm_indices_cpu = _stockham_perm_indices(log4n, B_pad, n)

    re = pad_re if use_sim else pad_re.to(device)
    im = pad_im if use_sim else pad_im.to(device)

    if not use_sim:
        # Transfer all per-stage twiddles and permutation indices to XLA device.
        # Each entry is an independent tensor — NEFF cache hit guaranteed.
        twiddles = [(tw_r.to(device), tw_i.to(device)) for tw_r, tw_i in twiddles_cpu]
        perm_indices = [(p.to(device), u.to(device)) for p, u in perm_indices_cpu]
    else:
        twiddles = twiddles_cpu
        perm_indices = perm_indices_cpu

    for s in range(log4n):
        pack_idx, unpack_idx = perm_indices[s]
        tw_r, tw_i = twiddles[s]

        # Pack: replace reshape+permute+contiguous+reshape with a single gather.
        # Reduces per-stage XLA graph from 4 ops to 1 (Thread C phase 1).
        re_groups = re.view(-1)[pack_idx].reshape(total_groups, 4)
        im_groups = im.view(-1)[pack_idx].reshape(total_groups, 4)

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

        # Unpack: single gather, replaces reshape+permute+contiguous+reshape.
        re = out_re.view(-1)[unpack_idx].reshape(B_pad, n)
        im = out_im.view(-1)[unpack_idx].reshape(B_pad, n)

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

    # Bench-toggle: force Stockham ahead of any threshold check. Only use
    # inside benchmark timing blocks; see note at _FORCE_STOCKHAM definition.
    if _FORCE_STOCKHAM and precision != "kahan":
        return _fft_via_stockham_nki(x, inverse)

    if n <= _DFT_GEMM_THRESHOLD and precision != "kahan":
        return _fft_via_gemm(x, inverse)

    # Twiddle precompute (SHA a74b697, 2026-04-17) closed the launch-count gap:
    # Stockham radix-4 is 6–9% faster than butterfly at all power-of-four N
    # (hardware-validated, trn1, SDK 2.29). Auto-dispatch for N > DFT-GEMM
    # threshold where the input is a power of four.
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
