"""
FFT core algorithms.

Iterative radix-2 Cooley-Tukey (decimation-in-time) for power-of-2 sizes.
Bluestein chirp-z for arbitrary sizes.
All operations on split real/imaginary tensors for Trainium compatibility.
"""

from __future__ import annotations

import functools
import math
import os
import warnings

import numpy as np
import torch

from .complex import ComplexTensor
from .nki.dispatch import _to_host, _use_nki
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
    indices = _bit_reverse_indices(n, log2n).to(x.real.device)
    re = x.real[..., indices].clone()
    im = x.imag[..., indices].clone()

    # Precompute twiddle factors for all stages
    # W_N^k = exp(sign * 2πi * k / N)
    for s in range(log2n):
        m = 1 << (s + 1)  # Butterfly group size
        half = m >> 1  # Half group

        # Twiddle: exp(sign * 2πi * k / m) for k = 0..half-1
        angles = sign * 2.0 * math.pi * torch.arange(half, dtype=torch.float64) / m
        tw_re = torch.cos(angles).to(dtype=re.dtype, device=re.device)
        tw_im = torch.sin(angles).to(dtype=re.dtype, device=re.device)

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
#   Default 1024 (trnfft#82). Trn2 (SDK 2.33 / NKI 0.7.0, 2026-10-27), batch
#   32, device-resident warm median: single GEMM 0.26 ms (N=512) / 0.31 ms
#   (N=1024) vs four-step 1.23 / 2.04 ms (fused twiddle) vs pre-#82 Stockham
#   5.45 / 7.56 ms. Device rel err vs FP64: 1.35e-7 (N=512), 1.54e-7
#   (N=1024). The earlier 256 cap cited "2.2% rel err at N=1024" on trn1;
#   that was the pre-#81 FP32 twiddle build (angles ~2*pi*N rounded to
#   FP32), not the K=N PSUM accumulation, and is obsolete. At N=2048 the
#   single GEMM (1.08 ms) and four-step (1.20 ms) are close; at N=4096
#   four-step wins (2.18 vs 2.36 ms) and the N^2 W stops being cheap.
#
#   With the batch-fused four-step kernel enabled (the default, see
#   _FOUR_STEP_BATCH_FUSED) the single GEMM only serves N < 512
#   (_FOUR_STEP_FUSED_MIN); this threshold still bounds it otherwise.
#
#   The low-precision GEMM modes (bf16 / bf16_refined / ozaki / ozaki_hq)
#   keep their validated N <= 256 range via _LOWP_GEMM_MAX.
#
#   Env knob TRNFFT_DFT_GEMM_THRESHOLD (read at import; tests still
#   monkeypatch the module constant) lets a device A/B the single-GEMM path
#   against the four-step / Stockham paths. Setting it to 0 disables every
#   GEMM path (DFT-GEMM and four-step), which is what the "force
#   butterfly/Stockham" tests rely on.


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(f"{name}={raw!r} is not an integer; using {default}", stacklevel=2)
        return default


_DFT_GEMM_THRESHOLD = _env_int("TRNFFT_DFT_GEMM_THRESHOLD", 1024)


# Rows per Stockham kernel launch (trnfft#83). The stage kernels take a
# (total_groups, r) operand and the compiler unrolls their 128-row tile loop,
# so the kernel body grows with the row count. On Trn2 (SDK 2.33 / NKI 0.7.0)
# N=2^25 put [8388608, 4] (2^23 rows) through the radix-4 kernel and BIR
# emission failed ("body blob exceeds uint32 offset range"). N <= 2^24 passed:
# the largest launches there were 2^21 rows (radix-8 at N=2^24, radix-4 at
# N=2^23). Each row is one independent r-point butterfly group (its own
# twiddle row, no cross-row reads), so larger stages are split into row chunks
# of at most this many rows, launched in a Python loop and concatenated. The
# kernels tile rows in blocks of 128, so a chunked launch computes exactly what
# the unchunked launch computes for those rows. Env TRNFFT_MAX_ROWS_PER_LAUNCH,
# read per call; rounded down to a multiple of 128, minimum 128.
_MAX_ROWS_PER_LAUNCH = 1 << 21
_STOCKHAM_ROW_TILE = 128


def _max_rows_per_launch() -> int:
    cap = _env_int("TRNFFT_MAX_ROWS_PER_LAUNCH", _MAX_ROWS_PER_LAUNCH)
    return max(_STOCKHAM_ROW_TILE, (cap // _STOCKHAM_ROW_TILE) * _STOCKHAM_ROW_TILE)


def _plan_row_chunks(total_rows: int, max_rows: int) -> list[tuple[int, int]]:
    """Split ``total_rows`` into ``[start, stop)`` launches of <= ``max_rows``.

    ``max_rows`` is rounded down to a multiple of 128 (minimum 128) so every
    chunk boundary lands on a kernel row-tile boundary. Returns one chunk
    covering all rows when no split is needed.
    """
    if total_rows <= 0:
        raise ValueError(f"total_rows must be positive, got {total_rows}")
    cap = max(_STOCKHAM_ROW_TILE, (max_rows // _STOCKHAM_ROW_TILE) * _STOCKHAM_ROW_TILE)
    if total_rows <= cap:
        return [(0, total_rows)]
    return [(start, min(start + cap, total_rows)) for start in range(0, total_rows, cap)]


def _launch_stockham_stage(
    kernel, rows_re, rows_im, arg_c, arg_d, use_sim: bool, per_row_cd: bool, op_name: str
):
    """Run a Stockham stage kernel over ``(total_groups, r)`` operands.

    ``arg_c``/``arg_d`` are per-row twiddles (``per_row_cd=True``, sliced with
    the rows) or a shared matrix such as W_8 (passed whole to every launch).
    Launches are split per :func:`_plan_row_chunks` (trnfft#83). Each launch
    goes through :func:`trnfft.nki.dispatch._launch` (``trnfft::<op_name>`` on
    the native path, ``kernel`` on the XLA path); operands are already on the
    kernel device, so nothing is copied.
    """
    from .nki.dispatch import _launch, _pe_transpose_enabled

    if op_name == "stockham_radix8_w8" and not _pe_transpose_enabled():
        # TRNFFT_PE_TRANSPOSE=0: the load_transpose2d W_8 kernel (trnfft#97).
        from .nki.stockham import stockham_radix8_w8_t2d_kernel

        kernel, op_name = stockham_radix8_w8_t2d_kernel, "stockham_radix8_w8_t2d"

    def _run(a, b, c, d):
        if use_sim:
            import nki as _nki

            o_re, o_im = _nki.simulate(kernel)(
                a.detach().cpu().numpy(),
                b.detach().cpu().numpy(),
                c.detach().cpu().numpy(),
                d.detach().cpu().numpy(),
            )
            return torch.from_numpy(np.asarray(o_re)), torch.from_numpy(np.asarray(o_im))
        return _launch(kernel, op_name, a, b, c, d)

    chunks = _plan_row_chunks(rows_re.shape[0], _max_rows_per_launch())
    if len(chunks) == 1:
        return _run(rows_re, rows_im, arg_c, arg_d)
    outs_re, outs_im = [], []
    for start, stop in chunks:
        c = arg_c[start:stop].contiguous() if per_row_cd else arg_c
        d = arg_d[start:stop].contiguous() if per_row_cd else arg_d
        o_re, o_im = _run(rows_re[start:stop].contiguous(), rows_im[start:stop].contiguous(), c, d)
        outs_re.append(o_re)
        outs_im.append(o_im)
    return torch.cat(outs_re, dim=0), torch.cat(outs_im, dim=0)


# Upper N for the bf16 / bf16_refined / ozaki / ozaki_hq GEMM modes, capped
# further by _DFT_GEMM_THRESHOLD. Their error bounds are characterised at
# N <= 256 only; above that they fall through to the FP32 single-GEMM /
# four-step paths.
_LOWP_GEMM_MAX = 256

# Four-step (Bailey) FFT-as-two-GEMMs (trnfft#82). For power-of-2 N with
# _DFT_GEMM_THRESHOLD < N <= _FOUR_STEP_MAX, N = N1*N2 (N1, N2 <= 64 at
# N=4096) is computed as GEMM(x, W_N1) -> twiddle -> GEMM(., W_N2): two
# Tensor-engine launches with K = N1, N2 instead of 3-4 Stockham kernel
# launches plus ~26-38 inter-stage permute/twiddle device ops and per-call
# twiddle H2D copies. With the default threshold it covers N=2048/4096:
# Trn2 1.20 / 2.18 ms (fused twiddle) vs Stockham 8.22 / 10.25 ms, device rel
# err 1.2e-7-1.6e-7. Env TRNFFT_FOUR_STEP_MAX (0 -> pre-#82 Stockham dispatch).
_FOUR_STEP_MAX = _env_int("TRNFFT_FOUR_STEP_MAX", 4096)

# Fuse the four-step twiddle multiply into one NKI complex-mul kernel launch
# (``_complex_mul_kernel``) instead of 6 elementwise device ops. Default on
# (trnfft#82): validated on Trn2, same accuracy, 1.67 -> 1.23 ms (N=512),
# 2.43 -> 2.18 ms (N=4096). Env TRNFFT_FOUR_STEP_FUSED_TWIDDLE=0 to disable.
_FOUR_STEP_FUSED_TWIDDLE = os.environ.get("TRNFFT_FOUR_STEP_FUSED_TWIDDLE", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Batch-fused four-step kernel (trnfft#82 follow-up): the whole four-step
# transform for a group of rows in one NKI launch (``four_step_fft_kernel``),
# with no transposes and one HBM read and write of the batch, instead of
# 2 GEMM launches + 1 twiddle launch + 6 transpose copies over the full batch.
# Trn2 (SDK 2.33 / NKI 0.7.0), N=4096, device-resident warm median: B=1024
# 10.39 -> 0.89 ms, B=128 3.41 -> 0.50 ms, rel err unchanged (1.58e-7).
# It also replaces the single DFT-GEMM from _FOUR_STEP_FUSED_MIN up: at
# N=512/1024, B=1024 it measured 0.37/0.49 ms vs 0.51/0.86 ms with lower
# error (1.22e-7/1.41e-7 vs 1.36e-7/1.54e-7); at N=256 the single GEMM stays
# ahead at large B. Env TRNFFT_FOUR_STEP_BATCH_FUSED=0 restores the unfused
# driver and the single GEMM up to _DFT_GEMM_THRESHOLD for A/B.
_FOUR_STEP_BATCH_FUSED = os.environ.get("TRNFFT_FOUR_STEP_BATCH_FUSED", "1").lower() in (
    "1",
    "true",
    "yes",
)
_FOUR_STEP_FUSED_MIN = _env_int("TRNFFT_FOUR_STEP_FUSED_MIN", 512)

# Rows per fused four-step launch. The kernel body is unrolled per row group,
# so the first-call compile grows with B (Trn2, N=4096: 3 s at B=1024, 17 s at
# B=8192); larger batches are split into launches of at most this many rows,
# which share one compiled shape. Throughput is linear in B either way
# (B=4096: 2.48 ms, B=8192: 4.73 ms in one launch).
_FOUR_STEP_FUSED_MAX_ROWS = _env_int("TRNFFT_FOUR_STEP_FUSED_MAX_ROWS", 4096)

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

# Bench toggle for the 2-level Ozaki path (v0.19).
_FORCE_OZAKI_HQ = False

# Hardware capability flag for Ozaki product precision.
#
# Ozaki precision improvement requires BF16 inputs to produce FP32-precision
# products before PSUM accumulation (i.e., nc_matmul computes W*x in FP32).
# trn1 does NOT provide this — BF16 products are rounded to BF16 before PSUM,
# leaving no signal for the correction terms to capture. Measured result on
# trn1: ozaki ≈ ozaki_hq ≈ bf16 (~1.7e-3 rel error at N=64).
# trn2 (SDK 2.33 / NKI 0.7.0, spike B5, 2026-10-27): ozaki/ozaki_hq matched the
# CPU emulation to two digits, i.e. products were not the limit. The remaining
# error was the FP32 twiddle build, fixed in trnfft#81.
#
# trn2/trn3 may provide FP32 product precision via TF32/MXFP8 tensor cores
# (same gap NVIDIA TF32 exploits). Run TestOzakiHQCharacterization to verify,
# then call set_ozaki_product_precision_verified(True) to enable.
#
# False  → emit RuntimeWarning, fall back to precision="bf16"
# True   → proceed without warning
_OZAKI_PRODUCT_PRECISION_VERIFIED = False


def set_ozaki_product_precision_verified(verified: bool = True) -> None:
    """Declare that nc_matmul on this hardware provides FP32 product precision.

    Call after running TestOzakiHQCharacterization and confirming ozaki_err
    is significantly better than bf16_err. Suppresses the trn1 warning and
    enables the ozaki/ozaki_hq precision modes without fallback.

    On trn1 (SDK 2.29.0): ozaki ≈ bf16 ≈ 1.7e-3. Do not set True on trn1.
    On trn2+: rerun characterization to verify before setting True.
    """
    global _OZAKI_PRODUCT_PRECISION_VERIFIED
    _OZAKI_PRODUCT_PRECISION_VERIFIED = verified


def _ozaki_or_fallback(x: ComplexTensor, inverse: bool, hq: bool) -> ComplexTensor:
    """Dispatch to Ozaki path if hardware is verified, else warn and use BF16."""
    if not _OZAKI_PRODUCT_PRECISION_VERIFIED:
        warnings.warn(
            "precision='ozaki_hq'"
            if hq
            else "precision='ozaki'"
            " requires nc_matmul to compute BF16×BF16 products at FP32 precision "
            "before PSUM accumulation. trn1 (SDK 2.29.0) does not provide this — "
            "measured ozaki ≈ bf16 ≈ 1.7e-3 rel error at N=64. "
            "Cost: " + ("6×" if hq else "3×") + " BF16 latency, no accuracy gain. "
            "Run TestOzakiHQCharacterization on your instance type and call "
            "trnfft.set_ozaki_product_precision_verified(True) if ozaki_err ≪ bf16_err. "
            "Falling back to precision='bf16'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fft_via_gemm_bf16(x, inverse)
    return _fft_via_ozaki_hq(x, inverse) if hq else _fft_via_ozaki(x, inverse)


def _dft_matrix_fp64(n: int, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of the N×N DFT matrix in FP64, angles reduced mod N.

    ``k*j`` is reduced modulo N in exact integer arithmetic before the angle is
    formed, so every angle lies in [0, 2π) and the FP64 cos/sin are correctly
    rounded to ~1e-16. Building W directly as ``cos(2π·k·j/N)`` in FP32 (the
    pre-#81 code) makes the angle up to ~2πN and rounds it to FP32, giving a
    twiddle error of ~N·1e-7 rel (5e-6 at N=64, 2e-5 at N=256). That error is
    a floor no Ozaki split can get under (trnfft#81).
    """
    sign = 1.0 if inverse else -1.0
    k = torch.arange(n, dtype=torch.int64)
    kj = (k.unsqueeze(1) * k.unsqueeze(0)) % n
    angles = sign * 2.0 * math.pi * kj.to(torch.float64) / n
    return torch.cos(angles), torch.sin(angles)


def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


@functools.lru_cache(maxsize=64)
def _dft_matrix_cached(
    n: int, inverse: bool, dtype: torch.dtype, device: torch.device, scale: float = 1.0
) -> ComplexTensor:
    """``_dft_matrix_fp64(n, inverse) * scale`` cast to ``dtype`` and resident on ``device``.

    Cached so the N x N FP64 cos/sin build and its host->device copy happen
    once per (n, direction, dtype, device) instead of on every call (trnfft#82).
    ``scale`` is used to fold the inverse 1/N into W; for power-of-2 N that is
    an exact power-of-2 scaling, so results are bit-identical to scaling after.
    """
    W_r, W_i = _dft_matrix_fp64(n, inverse)
    if scale != 1.0:
        W_r, W_i = W_r * scale, W_i * scale
    return ComplexTensor(W_r.to(dtype).to(device), W_i.to(dtype).to(device))


@functools.lru_cache(maxsize=64)
def _four_step_twiddle_cached(
    n1: int, n2: int, inverse: bool, dtype: torch.dtype, device: torch.device, rows: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Four-step twiddle ``T[n2, k1] = exp(sign*2*pi*i*n2*k1/N)``, (n2, n1), on ``device``.

    Built in FP64 with ``n2*k1`` reduced mod N (same recipe as
    ``_dft_matrix_fp64``), then cast once. With ``rows > 0`` the twiddle is
    tiled to a contiguous ``(rows * n2, n1)`` tensor (the fused complex-mul
    kernel takes equal shapes, no broadcast).
    """
    n = n1 * n2
    sign = 1.0 if inverse else -1.0
    a = torch.arange(n2, dtype=torch.int64).unsqueeze(1)
    b = torch.arange(n1, dtype=torch.int64).unsqueeze(0)
    ang = sign * 2.0 * math.pi * ((a * b) % n).to(torch.float64) / n
    t_re, t_im = torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)
    if rows > 0:
        t_re = t_re.repeat(rows, 1)
        t_im = t_im.repeat(rows, 1)
    return t_re.to(device), t_im.to(device)


def _four_step_split(n: int) -> tuple[int, int]:
    """N = N1 * N2 with N1 >= N2, both powers of 2 (N1 = N2 or 2*N2)."""
    log2n = int(math.log2(n))
    n2 = 1 << (log2n // 2)
    return n // n2, n2


def _use_fused_four_step(x: ComplexTensor, n: int) -> bool:
    """Route ``n`` to :func:`_fft_via_four_step_fused` (FP32, power of 2, in range)."""
    return (
        _FOUR_STEP_BATCH_FUSED
        and _DFT_GEMM_THRESHOLD > 0
        and max(_FOUR_STEP_FUSED_MIN, 4) <= n <= _FOUR_STEP_MAX
        and _is_power_of_two(n)
        and x.real.dtype == torch.float32
        and _four_step_fused_fits(n)
    )


def _four_step_fused_fits(n: int) -> bool:
    """``four_step_fft_kernel`` stacks re/im of N1 on 128 partitions: N1, N2 <= 64, N <= 4096."""
    n1, n2 = _four_step_split(n)
    return n1 <= 64 and n2 <= 64


@functools.lru_cache(maxsize=64)
def _four_step_fused_consts(
    n1: int,
    n2: int,
    inverse: bool,
    dtype: torch.dtype,
    device: torch.device,
    pack: int,
    packs: int,
) -> tuple[torch.Tensor, ...]:
    """Constant operands of ``four_step_fft_kernel``, resident on ``device``.

    ``(W1_blk, T_re, T_im, W2_re, W2_im)``. ``W1_blk`` is the real block form
    ``[[W1r, W1i], [-W1i, W1r]]`` of W_N1. ``W2`` is block-diagonal with
    ``pack`` copies of W_N2 (1/N folded in for the inverse, as in the unfused
    driver). ``T`` is ``T[n2, k1]`` tiled ``pack`` times along the partition
    axis and ``packs`` times along the free axis, ``(pack*N2, packs*N1)``.
    """
    n = n1 * n2
    cpu = torch.device("cpu")
    W1 = _dft_matrix_cached(n1, inverse, dtype, cpu)
    W2 = _dft_matrix_cached(n2, inverse, dtype, cpu, (1.0 / n) if inverse else 1.0)
    t_re, t_im = _four_step_twiddle_cached(n1, n2, inverse, dtype, cpu)
    w1_blk = torch.cat(
        [torch.cat([W1.real, W1.imag], dim=1), torch.cat([-W1.imag, W1.real], dim=1)], dim=0
    )
    out = (
        w1_blk,
        t_re.repeat(pack, packs),
        t_im.repeat(pack, packs),
        torch.block_diag(*[W2.real] * pack),
        torch.block_diag(*[W2.imag] * pack),
    )
    return tuple(t.contiguous().to(device) for t in out)


def _fft_via_four_step_fused(
    x: ComplexTensor, inverse: bool, max_rows: int | None = None
) -> ComplexTensor:
    """Four-step FFT for the whole batch in one NKI launch (trnfft#82 follow-up).

    Same math and constants as :func:`_fft_via_four_step`, but
    ``four_step_fft_kernel`` keeps each group of rows in SBUF/PSUM between the
    two GEMMs and the twiddle and reads the input in its natural layout, so
    there are no transposes, no batch padding and no intermediate HBM tensors:
    one launch, one read and one write of the (B, N) batch. Batches above
    ``_FOUR_STEP_FUSED_MAX_ROWS`` rows are split into several launches.

    On the native path a host batch is moved to the kernel device once, the
    launches run on device-resident slices, and the result returns to the
    host once (trnfft#94). Before, each launch's ``_launch`` copied its slice
    in and its result out separately, and the slices were concatenated on the
    host: B=16384, N=4096 from the host took 368 ms against 213 ms when moved
    to the device by the caller.
    """
    from .nki import four_step as _fs
    from .nki.dispatch import _is_native, _kernel_device, _launch, _simulate_kernel, _use_simulator

    # Looked up at call time: the kernel exists only when nki is importable
    # (the CPU dispatch tests install a stand-in).
    kernel = getattr(_fs, "four_step_fft_kernel", None)

    n = x.shape[-1]
    n1, n2 = _four_step_split(n)
    orig_shape = x.real.shape
    orig_device = x.real.device
    re = x.real.reshape(-1, n).contiguous()
    im = x.imag.reshape(-1, n).contiguous()
    if not _use_simulator() and _is_native() and orig_device.type != "neuron":
        dev = _kernel_device()
        re, im = re.to(dev), im.to(dev)

    def _run(r, i):
        pack = _fs.four_step_pack_rows(n2, r.shape[0])
        packs = _fs.four_step_group_packs(n1, r.shape[0], pack)
        consts = _four_step_fused_consts(n1, n2, inverse, r.dtype, r.device, pack, packs)
        if _use_simulator():
            return _simulate_kernel(kernel, r, i, *consts)
        return _launch(kernel, "four_step_fft", r, i, *consts)

    rows = re.shape[0]
    cap = max(1, _FOUR_STEP_FUSED_MAX_ROWS if max_rows is None else max_rows)
    if rows <= cap:
        o_re, o_im = _run(re, im)
    else:
        outs = [_run(re[s : s + cap], im[s : s + cap]) for s in range(0, rows, cap)]
        o_re = torch.cat([o[0] for o in outs], dim=0)
        o_im = torch.cat([o[1] for o in outs], dim=0)
        del outs
    o_re = _to_host(o_re.reshape(orig_shape), orig_device)
    o_im = _to_host(o_im.reshape(orig_shape), orig_device)
    return ComplexTensor(o_re, o_im)


# Six-step (Bailey) FFT for N above the fused four-step range (trnfft#54).
# N = N1 * N2 (balanced powers of 2) is computed as N1 row FFTs of length N2,
# a twiddle multiply, and N2 row FFTs of length N1, with three transposes on
# the device. Each row pass is one batched transform, so it runs on the
# batch-fused four-step kernel (N2 <= 4096) or the single DFT-GEMM, instead
# of the 8-13 row-chunked Stockham stages. Lengths above _FOUR_STEP_MAX
# recurse.
#
# Default for power-of-2 FP32 N >= _SIX_STEP_MIN (trnfft#94). On Trn2 (SDK
# 2.33, one logical core) it beat the Stockham path at every N and batch
# measured from 2^13 to 2^26, warm and on the first call, from device and
# host input, with 10x lower error: see CHANGELOG. TRNFFT_SIX_STEP (read per
# call): unset -> default, 0 -> never (the Stockham path), 1 -> every N above
# _FOUR_STEP_MAX. TRNFFT_SIX_STEP_MIN moves the default threshold.
_SIX_STEP_MIN = _env_int("TRNFFT_SIX_STEP_MIN", _FOUR_STEP_MAX * 2)


def _six_step_mode() -> str:
    """``"on"`` / ``"off"`` from ``TRNFFT_SIX_STEP``, else ``"auto"``."""
    raw = os.environ.get("TRNFFT_SIX_STEP", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return "on"
    if raw in ("0", "false", "no", "off"):
        return "off"
    return "auto"


def _use_six_step(x: ComplexTensor, n: int, precision: str = "fast") -> bool:
    """Route ``n`` to :func:`_fft_via_six_step` (FP32, power of 2, N > _FOUR_STEP_MAX).

    On by default from ``_SIX_STEP_MIN``; ``TRNFFT_SIX_STEP=0`` turns it off
    and ``=1`` turns it on for every N above ``_FOUR_STEP_MAX``.
    """
    mode = _six_step_mode()
    if mode == "off" or (mode == "auto" and n < _SIX_STEP_MIN):
        return False
    return (
        precision in ("fast", "kahan")
        and _DFT_GEMM_THRESHOLD > 0
        and n > _FOUR_STEP_MAX >= 4
        and _is_power_of_two(n)
        and x.real.dtype == torch.float32
    )


# Rows per launch for a six-step pass that lands on the single DFT-GEMM
# (length <= _DFT_GEMM_THRESHOLD). That kernel unrolls its 128-row M tiles,
# so a 2^19-row pass is one very large kernel to compile; passes are split
# into launches of at most this many rows instead. The fused four-step
# kernel already splits at _FOUR_STEP_FUSED_MAX_ROWS.
_SIX_STEP_GEMM_MAX_ROWS = _env_int("TRNFFT_SIX_STEP_GEMM_MAX_ROWS", 1 << 15)

# Six-step passes use the fused four-step kernel from this length up, below
# the _FOUR_STEP_FUSED_MIN = 512 of the normal dispatch. On Trn2 the passes
# have 10^4-10^5 rows, where the fused kernel beats the single DFT-GEMM:
# B=256 N=2^16 (256-point passes) 47 -> 19 ms, N=2^26 (128-point passes)
# 256 -> 151 ms, same accuracy. Env TRNFFT_SIX_STEP_FUSED_MIN.
_SIX_STEP_FUSED_MIN = _env_int("TRNFFT_SIX_STEP_FUSED_MIN", 128)


def _six_step_fused_rows(length: int) -> int:
    """Rows per fused launch for a six-step pass of ``length``.

    The fused kernel unrolls one loop body per group of P*Q rows (P rows per
    stationary operand, Q packs per PSUM tile), which is 16 rows at N=4096
    and up to 512 at N=128. Scaling _FOUR_STEP_FUSED_MAX_ROWS by P*Q/16 keeps
    the unrolled body, and so the compile, the same size as at N=4096 while
    small-N passes need far fewer launches.
    """
    from .nki.four_step import FOUR_STEP_FREE_MAX

    n1, n2 = _four_step_split(length)
    pq = max(1, 128 // n2) * max(1, FOUR_STEP_FREE_MAX // n1)
    return max(_FOUR_STEP_FUSED_MAX_ROWS, _FOUR_STEP_FUSED_MAX_ROWS * pq // 16)


def _six_step_split(n: int) -> tuple[int, int]:
    """N = N1 * N2, powers of 2.

    Up to _FOUR_STEP_MAX^2 (2^24) the split is balanced (N1 = N2 or 2 * N2),
    so both passes are single-level. Above it N1 = _FOUR_STEP_MAX, one pass
    on the fused kernel, and N2 = N / N1 recurses.
    """
    if n <= _FOUR_STEP_MAX * _FOUR_STEP_MAX:
        return _four_step_split(n)
    return _FOUR_STEP_MAX, n // _FOUR_STEP_MAX


def _six_step_pass(re: torch.Tensor, im: torch.Tensor, inverse: bool, precision: str):
    """Row FFTs of a (rows, L) pair on its device; L <= _FOUR_STEP_MAX or recursive."""
    rows = re.shape[0]
    if rows > 128 and rows % 128:
        # The single DFT-GEMM tiles rows by 128 with no tail tile.
        pad = torch.zeros(128 - rows % 128, re.shape[1], dtype=re.dtype, device=re.device)
        re, im = torch.cat([re, pad]), torch.cat([im, pad])

    def run(r, i):
        x = ComplexTensor(r, i)
        if _use_nki():
            return _cooley_tukey_nki_nograd(x, inverse=inverse, precision=precision)
        return fft_core(x, inverse=inverse, precision=precision)  # CPU reference (tests)

    length = re.shape[1]
    if (
        _use_nki()
        and _FOUR_STEP_BATCH_FUSED
        and max(_SIX_STEP_FUSED_MIN, 4) <= length <= _FOUR_STEP_MAX
        and _four_step_fused_fits(length)
    ):
        out = _fft_via_four_step_fused(ComplexTensor(re, im), inverse, _six_step_fused_rows(length))
        return out.real[:rows], out.imag[:rows]
    cap = max(128, _SIX_STEP_GEMM_MAX_ROWS // 128 * 128)
    if length <= _DFT_GEMM_THRESHOLD and re.shape[0] > cap:
        outs = [run(re[s : s + cap], im[s : s + cap]) for s in range(0, re.shape[0], cap)]
        o_re = torch.cat([o.real for o in outs])
        o_im = torch.cat([o.imag for o in outs])
    else:
        out = run(re, im)
        o_re, o_im = out.real, out.imag
    return o_re[:rows], o_im[:rows]


def _six_step_pass1_tn_ok(n1: int, n2: int) -> bool:
    """Pass 1 (rows of length N2) runs on ``_complex_gemm_tn_kernel`` (trnfft#97).

    That is where ``_six_step_pass`` would take the single DFT-GEMM: NKI on,
    N2 below the fused range and within the GEMM threshold. The kernel's
    tiles need N1 <= 128 or a multiple of 128, and N2 <= 128 or a multiple
    of 128. ``TRNFFT_SIX_STEP_TN=0`` (read per call) keeps the transpose and
    the ``load_transpose2d`` GEMM.
    """
    if os.environ.get("TRNFFT_SIX_STEP_TN", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    fused = _FOUR_STEP_BATCH_FUSED and max(_SIX_STEP_FUSED_MIN, 4) <= n2 <= _FOUR_STEP_MAX
    return (
        _use_nki()
        and not (fused and _four_step_fused_fits(n2))
        and n2 <= _DFT_GEMM_THRESHOLD
        and (n1 <= 128 or n1 % 128 == 0)
        and (n2 <= 128 or n2 % 128 == 0)
    )


def _six_step_pass1_tn(re, im, B: int, n1: int, n2: int, inverse: bool):
    """Pass 1 straight from the (B, N2, N1) view: rows (b, j1) of length N2.

    Block b of ``re.reshape(B*N2, N1)`` is ``x[b]`` with j2 on rows, which is
    the K-major left operand ``_complex_gemm_tn_kernel`` takes, so neither
    the transpose to (B, N1, N2) nor a transposed load is needed. Bit-identical
    to transposing and running ``_fft_via_gemm`` (same W, tiles and PSUM
    order). Launches hold at most ``_SIX_STEP_GEMM_MAX_ROWS`` output rows.
    """
    from .nki import dispatch as _d
    from .nki.dispatch import _launch, _simulate_kernel, _use_simulator

    # Looked up at call time: the kernel exists only when nki is importable.
    kernel = getattr(_d, "_complex_gemm_tn_kernel", None)
    fold = inverse  # N2 is a power of 2: the 1/N2 folds into W exactly
    W = _dft_matrix_cached(n2, inverse, re.dtype, re.device, (1.0 / n2) if fold else 1.0)
    at_re = re.reshape(B * n2, n1)
    at_im = im.reshape(B * n2, n1)
    per = max(1, _SIX_STEP_GEMM_MAX_ROWS // n1)

    def run(s, e):
        a_r, a_i = at_re[s * n2 : e * n2], at_im[s * n2 : e * n2]
        if _use_simulator():
            return _simulate_kernel(kernel, a_r, a_i, W.real, W.imag)
        return _launch(kernel, "complex_gemm_tn", a_r, a_i, W.real, W.imag)

    outs = [run(s, min(s + per, B)) for s in range(0, B, per)]
    if len(outs) == 1:
        return outs[0]
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def _fft_via_six_step(x: ComplexTensor, inverse: bool, precision: str = "fast") -> ComplexTensor:
    """Six-step FFT: transpose, N1 row FFTs of length N2, twiddle, transpose,
    N2 row FFTs of length N1, transpose (trnfft#54).

    With input index ``j = j1 + N1*j2`` and output index ``k = k2 + N2*k1``::

        X[k2 + N2*k1] = sum_j1 W_N1^(j1 k1) W_N^(j1 k2) sum_j2 x[j1 + N1 j2] W_N2^(j2 k2)

    The (B, N2, N1) view of x is transposed to (B, N1, N2) so the inner sum
    is a batch of B*N1 row FFTs; the twiddle ``W_N^(j1 k2)`` (built in FP64,
    cast once, cached per device) multiplies the (N1, N2) result; a second
    transpose gives B*N2 rows of length N1; the last transpose puts
    ``k = k1*N2 + k2`` in natural order. The inverse uses the conjugate
    twiddle and the inner transforms' 1/N1 and 1/N2, so 1/N overall.

    A host input is moved to the kernel device once and the result returned
    to the host, so the intermediates and transposes stay on the device.
    """
    from .nki.dispatch import _is_native, _kernel_device, _use_simulator

    n = x.shape[-1]
    n1, n2 = _six_step_split(n)
    orig_shape = x.real.shape
    orig_device = x.real.device
    re = x.real.reshape(-1, n)
    im = x.imag.reshape(-1, n)
    if not _use_simulator() and _is_native() and orig_device.type != "neuron":
        dev = _kernel_device()
        re, im = re.to(dev), im.to(dev)
    B = re.shape[0]
    dtype, device = re.dtype, re.device

    # x[b, j2, j1] -> rows (b, j1), length N2.
    if _six_step_pass1_tn_ok(n1, n2):
        y_re, y_im = _six_step_pass1_tn(re.contiguous(), im.contiguous(), B, n1, n2, inverse)
    else:
        a_re = re.reshape(B, n2, n1).transpose(1, 2).contiguous().reshape(B * n1, n2)
        a_im = im.reshape(B, n2, n1).transpose(1, 2).contiguous().reshape(B * n1, n2)
        y_re, y_im = _six_step_pass(a_re, a_im, inverse, precision)
        del a_re, a_im

    # Twiddle W_N^(j1 k2), (N1, N2), broadcast over the batch.
    t_re, t_im = _four_step_twiddle_cached(n2, n1, inverse, dtype, device)
    y_re = y_re.reshape(B, n1, n2)
    y_im = y_im.reshape(B, n1, n2)
    z_re = y_re * t_re - y_im * t_im
    z_im = y_re * t_im + y_im * t_re
    del y_re, y_im

    # (b, j1, k2) -> rows (b, k2), length N1.
    b_re = z_re.transpose(1, 2).contiguous().reshape(B * n2, n1)
    b_im = z_im.transpose(1, 2).contiguous().reshape(B * n2, n1)
    del z_re, z_im
    o_re, o_im = _six_step_pass(b_re, b_im, inverse, precision)
    del b_re, b_im

    # (b, k2, k1) -> (b, k1, k2): k = k1*N2 + k2.
    X_re = o_re.reshape(B, n2, n1).transpose(1, 2).contiguous()
    X_im = o_im.reshape(B, n2, n1).transpose(1, 2).contiguous()
    X_re = _to_host(X_re.reshape(orig_shape), orig_device)
    X_im = _to_host(X_im.reshape(orig_shape), orig_device)
    return ComplexTensor(X_re, X_im)


def _fft_via_four_step(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Four-step (Bailey) FFT as two complex GEMMs on the Tensor engine (trnfft#82).

    With N = N1*N2, input index n = N2*n1 + n2 and output index k = k1 + N1*k2::

        Y[b, n2, k1] = sum_n1 x[b, N2*n1 + n2] * W_N1[n1, k1]      (GEMM 1, K = N1)
        Y[b, n2, k1] *= T[n2, k1] = W_N^(n2*k1)                    (twiddle)
        Z[b, k1, k2] = sum_n2 Y[b, n2, k1] * W_N2[n2, k2]          (GEMM 2, K = N2)
        X[b, k1 + N1*k2] = Z[b, k1, k2]

    Two kernel launches plus three transposes and the twiddle multiply, versus
    one launch per Stockham stage plus per-stage pack/unpack permutes and
    twiddle multiplies. Each GEMM accumulates only K = N1 or N2 (<= 64 at
    N=4096) terms, so the FP32 error stays at the small-N DFT-GEMM level.
    W_N1, W_N2 and T are cached on the input's device; for the inverse the
    1/N is folded into W_N2 (exact for power-of-2 N).
    """
    from .nki.dispatch import complex_gemm, complex_mask_apply

    n = x.shape[-1]
    n1, n2 = _four_step_split(n)
    orig_shape = x.real.shape
    dtype, device = x.real.dtype, x.real.device

    re = x.real.reshape(-1, n)
    im = x.imag.reshape(-1, n)
    B = re.shape[0]
    # _complex_gemm_kernel tiles M by min(M, 128) with no tail tile: pad the
    # batch so B*N1 and B*N2 are multiples of 128 whenever they exceed 128.
    B_pad = B
    if (B * n2 > 128 and (B * n2) % 128) or (B * n1 > 128 and (B * n1) % 128):
        q = 128 // min(n1, n2, 128)
        B_pad = ((B + q - 1) // q) * q
    if B_pad != B:
        pad = torch.zeros(B_pad - B, n, dtype=dtype, device=device)
        re = torch.cat([re, pad], dim=0)
        im = torch.cat([im, pad], dim=0)

    W1 = _dft_matrix_cached(n1, inverse, dtype, device)
    W2 = _dft_matrix_cached(n2, inverse, dtype, device, (1.0 / n) if inverse else 1.0)

    # (B, N1, N2) -> (B, N2, N1): rows (b, n2), contraction axis n1.
    a_re = re.reshape(B_pad, n1, n2).transpose(1, 2).contiguous().reshape(B_pad * n2, n1)
    a_im = im.reshape(B_pad, n1, n2).transpose(1, 2).contiguous().reshape(B_pad * n2, n1)
    Y = complex_gemm(ComplexTensor(a_re, a_im), W1)

    if _FOUR_STEP_FUSED_TWIDDLE:
        T = ComplexTensor(*_four_step_twiddle_cached(n1, n2, inverse, dtype, device, B_pad))
        Y = complex_mask_apply(T, Y)
        y_re = Y.real.reshape(B_pad, n2, n1)
        y_im = Y.imag.reshape(B_pad, n2, n1)
    else:
        t_re, t_im = _four_step_twiddle_cached(n1, n2, inverse, dtype, device)
        yr = Y.real.reshape(B_pad, n2, n1)
        yi = Y.imag.reshape(B_pad, n2, n1)
        y_re = yr * t_re - yi * t_im
        y_im = yr * t_im + yi * t_re

    # (B, N2, N1) -> (B, N1, N2): rows (b, k1), contraction axis n2.
    b_re = y_re.transpose(1, 2).contiguous().reshape(B_pad * n1, n2)
    b_im = y_im.transpose(1, 2).contiguous().reshape(B_pad * n1, n2)
    Z = complex_gemm(ComplexTensor(b_re, b_im), W2)

    # (B, k1, k2) -> (B, k2, k1) -> natural order k = k1 + N1*k2.
    X_re = Z.real.reshape(B_pad, n1, n2).transpose(1, 2).contiguous().reshape(B_pad, n)
    X_im = Z.imag.reshape(B_pad, n1, n2).transpose(1, 2).contiguous().reshape(B_pad, n)
    if B_pad != B:
        X_re, X_im = X_re[:B], X_im[:B]
    return ComplexTensor(X_re.reshape(orig_shape), X_im.reshape(orig_shape))


@functools.lru_cache(maxsize=64)
def _dft_matrix_blockdiag_cached(
    n: int, g: int, inverse: bool, dtype: torch.dtype, device: torch.device, scale: float = 1.0
) -> ComplexTensor:
    """``g`` copies of ``_dft_matrix_cached(n, ...)`` on the diagonal, (g*n, g*n)."""
    W = _dft_matrix_cached(n, inverse, dtype, torch.device("cpu"), scale)
    W_r = torch.block_diag(*[W.real] * g)
    W_i = torch.block_diag(*[W.imag] * g)
    return ComplexTensor(W_r.to(device), W_i.to(device))


def _dft_gemm_pack(rows: int, n: int) -> int:
    """Rows of length ``n`` packed side by side into one GEMM row (trnfft#97).

    ``_complex_gemm_kernel`` transposes each (128, K) tile of its left operand
    on the Tensor engine. With K = n < 128 most of each transpose and matmul
    is idle, and at 32768 x 8 the per-tile work made the kernel 1.5x slower
    than the ``load_transpose2d`` one. Viewing x as (rows/g, g*n) with a
    block-diagonal (g*n, g*n) W does g row FFTs per GEMM row: g times fewer
    tiles, loads, transposes and matmuls. The extra terms are exact zero
    products, so each output is the same sum as without packing.

    g is the largest power of two with g*n <= 128 that divides ``rows`` and
    keeps rows/g <= 128 or a multiple of 128 (the kernel's M tiling). 1 when
    the packing does not apply: n >= 128, NKI off, or ``TRNFFT_PE_TRANSPOSE=0``
    (which keeps the unpacked ``load_transpose2d`` GEMM). ``TRNFFT_GEMM_PACK=0``
    (read per call) turns off the packing alone.
    """
    from .nki.dispatch import _pe_transpose_enabled

    if n >= 128 or not _use_nki() or not _pe_transpose_enabled():
        return 1
    if os.environ.get("TRNFFT_GEMM_PACK", "1").strip().lower() in ("0", "false", "no", "off"):
        return 1
    g = 128 // n if _is_power_of_two(n) else 1
    while g > 1:
        r = rows // g
        if rows % g == 0 and (r <= 128 or r % 128 == 0):
            return g
        g //= 2
    return 1


def _fft_via_gemm(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Compute FFT as a single complex matmul: X = W @ x (FP32, NKI path).

    Routes onto the Trainium Tensor engine via `complex_gemm`. PSUM
    accumulation on Trainium is always FP32; Trn2 measured 1.35e-7 / 1.54e-7
    relative error at N=512 / 1024 (default threshold 1024, trnfft#82). For FP64 accuracy,
    use ``precision="double"`` which routes to :func:`_fft_via_gemm_double`.

    Shape handling: flattens leading batch dims to 2D (B, N), does
    ``x @ W`` (W is the N×N DFT matrix), restores original shape.
    """
    from .nki.dispatch import complex_gemm

    n = x.shape[-1]
    orig_shape = x.real.shape
    x_re = x.real.reshape(-1, n).contiguous()
    x_im = x.imag.reshape(-1, n).contiguous()

    # Cached on the input's device (trnfft#82): no per-call N x N FP64 build
    # or host->device copy. For power-of-2 N the inverse 1/N is folded into W
    # (exact), saving the two scaling ops after the GEMM.
    fold = inverse and _is_power_of_two(n)
    scale = (1.0 / n) if fold else 1.0
    rows_in = x_re.shape[0]
    if _use_nki() and rows_in > 128 and rows_in % 128:
        # The kernel tiles M by 128 with no tail tile: rows past the last
        # full tile were never written (B=1000, N=64: rows 896-999 wrong).
        pad = torch.zeros(128 - rows_in % 128, n, dtype=x_re.dtype, device=x_re.device)
        x_re, x_im = torch.cat([x_re, pad]), torch.cat([x_im, pad])
    rows = x_re.shape[0]
    g = _dft_gemm_pack(rows, n)
    if g > 1:
        # (rows, n) -> (rows/g, g*n) is a free view; W becomes g copies of W
        # on the diagonal, so each packed row's g segments transform alone.
        W = _dft_matrix_blockdiag_cached(n, g, inverse, x_re.dtype, x_re.device, scale)
        X_p = complex_gemm(
            ComplexTensor(x_re.reshape(rows // g, g * n), x_im.reshape(rows // g, g * n)), W
        )
        X_2d = ComplexTensor(X_p.real.reshape(rows, n), X_p.imag.reshape(rows, n))
    else:
        W = _dft_matrix_cached(n, inverse, x_re.dtype, x_re.device, scale)
        X_2d = complex_gemm(ComplexTensor(x_re, x_im), W)
    X_re = X_2d.real[:rows_in].reshape(orig_shape)
    X_im = X_2d.imag[:rows_in].reshape(orig_shape)
    result = ComplexTensor(X_re, X_im)
    if inverse and not fold:
        result = result * (1.0 / n)
    return result


def _fft_host_fp64(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """FP64 FFT on the host, result on ``x``'s device in ``x``'s dtype.

    ``precision="double"`` above the FP64 DFT-GEMM range: the NKI kernels
    are FP32-only (#108), and an N^2 W does not scale, so this is the host's
    FP64 FFT, the same accuracy-first trade as :func:`_fft_via_gemm_double`.
    """
    dev, dt = x.real.device, x.real.dtype
    z = torch.complex(x.real.detach().cpu().double(), x.imag.detach().cpu().double())
    y = torch.fft.ifft(z) if inverse else torch.fft.fft(z)
    return ComplexTensor(y.real.to(dtype=dt, device=dev), y.imag.to(dtype=dt, device=dev))


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

    W = ComplexTensor(*_dft_matrix_fp64(n, inverse))

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

    BF16 Tensor Engine throughput is ≈2× FP32. Rounding W and x to BF16 sets
    the accuracy. The normwise relative error is about u_bf16 = 2^-9 and grows
    slowly with log N. Measured (CPU emulation, which Trn2 reproduces to 3 digits;
    max over 20 seeds): 1.9e-3 (N=2) to 2.16e-3 (N=256). Trn2 measured 2.06e-3 /
    2.09e-3 at N=128/256 (trnfft#86). Stated bound: 3e-3 for N ≤ 256. For
    ~5e-6, use :func:`_fft_iterative_refinement`, which costs an extra BF16 GEMM
    and an FP32 GEMM.

    On CPU (no NKI): falls back to ``complex_matmul`` with BF16 inputs — same
    dtype semantics, no throughput benefit but identical correctness test path.
    """
    from .nki.dispatch import complex_gemm_bf16

    n = x.shape[-1]
    orig_shape = x.real.shape

    # Cast x to BF16 for compute. W is built in FP32 then quantized to BF16:
    # computing angles in BF16 introduces large errors for high k*j products
    # (BF16 loses precision for integers > 128). Computing in FP32 and quantizing
    # W entries to BF16 gives the intended ~2e-3 (u_bf16) rel error regime.
    x_re = x.real.reshape(-1, n).to(torch.bfloat16).contiguous()
    x_im = x.imag.reshape(-1, n).to(torch.bfloat16).contiguous()

    # Quantize W to BF16 — this is the BF16 DFT-matrix approximation.
    W_r, W_i = _dft_matrix_fp64(n, inverse)
    W = ComplexTensor(W_r.bfloat16(), W_i.bfloat16())

    # complex_gemm_bf16 returns FP32 (PSUM not rounded to BF16).
    X_2d = complex_gemm_bf16(ComplexTensor(x_re, x_im), W)
    X_re = X_2d.real.reshape(orig_shape)
    X_im = X_2d.imag.reshape(orig_shape)
    result = ComplexTensor(X_re, X_im)
    if inverse:
        result = result * (1.0 / n)
    return result


def _fft_iterative_refinement(x: ComplexTensor, inverse: bool, steps: int = 1) -> ComplexTensor:
    """BF16 DFT-GEMM + iterative-refinement (IR) correction steps.

    Algorithm (forward FFT case, inverse analogous):

    .. code-block:: text

        X̂ = fft_bf16(x)                  # BF16 W and x, FP32 PSUM output
        for _ in range(steps):
            x_rec = ifft_fp32(X̂)          # FP32 DFT-GEMM (exact W from FP64)
            r = x_fp32 - x_rec            # FP32 residual, |r| ~ u_bf16·|x|
            δ = fft_bf16(r)               # BF16 FFT of the small residual
            X̂ = X̂ + δ

    The residual must be computed with the accurate operator. Write the BF16
    transform as W̃ = W + E with ‖E‖ ~ u_bf16·‖W‖. With an exact reconstruction,
    r = x − W⁻¹W̃x = −W⁻¹Ex and X̂ + W̃r = Wx − EW⁻¹Ex, so the O(u_bf16) term
    cancels and O(u_bf16²) is left. Before trnfft#86 the reconstruction used the
    BF16 inverse W̃' = W⁻¹ + E' as well. Then r = −(W⁻¹E + E'W)x, and the
    correction cancels Ex but leaves −WE'Wx, which is the same size: the error
    stays O(u_bf16). Trn2 measured
    1.95e-3 / 2.03e-3 / 2.08e-3 at N=64/128/256, the same as plain "bf16", and
    CPU emulation reproduces those values to 3 digits.

    Measured normwise relative error (CPU emulation of the BF16 kernel, which
    is exact for BF16×BF16 products with FP32 accumulation; batch 8 complex
    Gaussian, max over 20 seeds): 3.6e-6 (N=2) to 4.9e-6 (N=256), forward and
    inverse. Stated bound: 1e-5 for N ≤ 256 (``_LOWP_GEMM_MAX``). What remains
    is mostly the BF16 rounding of r: about u_bf16 times the ~2e-3 relative size
    of r.

    Cost per step: two BF16 GEMMs plus one FP32 GEMM. That is more than the
    single FP32 GEMM of "fast", which is also more accurate (1.5e-7 to 3e-7).
    This mode makes sense where the BF16 forward result is consumed anyway and
    a near-FP32 correction is wanted. It is not a faster route to FP32 accuracy.
    """
    xf = ComplexTensor(x.real.float(), x.imag.float())

    # BF16 FFT in the requested direction, FP32 output from PSUM.
    x_hat = _fft_via_gemm_bf16(xf, inverse)

    for _ in range(steps):
        # Reconstruct the input with the FP32 DFT-GEMM in the opposite direction.
        # This has to be accurate (see docstring): a BF16 reconstruction leaves the
        # error at O(u_bf16), and that was trnfft#86. _fft_via_gemm applies 1/N for
        # inverse=True.
        x_rec = _fft_via_gemm(x_hat, not inverse)

        # FP32 residual, small (~u_bf16 relative), so BF16 is enough to transform it.
        r = ComplexTensor(xf.real - x_rec.real, xf.imag - x_rec.imag)

        delta = _fft_via_gemm_bf16(r, inverse)
        x_hat = ComplexTensor(x_hat.real + delta.real, x_hat.imag + delta.imag)

    return x_hat


def _ozaki_split_bf16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split FP32 tensor into BF16 high + low parts (1-level ORO split)."""
    x_high = x.bfloat16()
    x_low = (x - x_high.to(x.dtype)).bfloat16()
    return x_high, x_low


def _ozaki_split_3way_bf16(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-way ORO split: x ≈ x_h1 + x_h2 + x_h3, all BF16, FP32 residuals between levels.

    This is the key constraint for 2-level Ozaki: the first residual must stay in FP32
    before the second BF16 quantisation. A BF16 value has zero remaining mantissa bits
    for a second split — the FP32 intermediate is what enables O(u_bf16^4) accuracy.

    Magnitudes: |x_h1| ~ |x|, |x_h2| ~ u_bf16·|x|, |x_h3| ~ u_bf16²·|x|.
    """
    x_h1 = x.bfloat16()
    r1 = x - x_h1.to(x.dtype)  # FP32/FP64 residual — NOT cast to BF16 here
    x_h2 = r1.bfloat16()
    x_h3 = (r1 - x_h2.to(x.dtype)).bfloat16()
    return x_h1, x_h2, x_h3


def _fft_via_ozaki(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Ozaki-scheme DFT-GEMM: 1-level BF16 split, 3 matmuls, FP32 accumulation.

    Decomposes W and x into BF16 high/low parts, calls _fft_via_gemm_bf16 once
    per cross-term (3 total), and sums the FP32 results. Uses _fft_via_gemm_bf16
    directly so each term takes exactly the same hardware path as the working
    BF16 benchmark.

    Expected accuracy: ~3.5e-6–4e-6 rel error at N=64–256 (dominated by the
    dropped W_l@x_l term and the W/x split residuals, all O(u_bf16^2)). W is built
    and split from FP64 (trnfft#81); the earlier FP32 W added ~N·1e-7 of twiddle
    error on top.
    Cost: 3× BF16 DFT-GEMM ≈ 2× FP32 DFT-GEMM.
    """
    n = x.shape[-1]
    orig_shape = x.real.shape

    # W in FP64 (angles reduced mod n) and split from FP64: an FP32 W carries
    # ~N·1e-7 twiddle error that the split cannot remove (trnfft#81).
    W_r_fp64, W_i_fp64 = _dft_matrix_fp64(n, inverse)

    # Ozaki splits (CPU → BF16)
    x_re_fp32 = x.real.reshape(-1, n).contiguous()
    x_im_fp32 = x.imag.reshape(-1, n).contiguous()
    x_r_h, x_r_l = _ozaki_split_bf16(x_re_fp32)
    x_i_h, x_i_l = _ozaki_split_bf16(x_im_fp32)
    W_r_h, W_r_l = _ozaki_split_bf16(W_r_fp64)
    W_i_h, W_i_l = _ozaki_split_bf16(W_i_fp64)

    def _term(xr, xi, wr, wi) -> ComplexTensor:
        """One BF16 DFT-GEMM term using the validated BF16 kernel path."""
        # Build a fake ComplexTensor carrying BF16 split parts, then call
        # _fft_via_gemm_bf16's kernel directly.  We replicate just the matmul
        # portion (no angle recomputation) via complex_gemm_bf16.
        from .nki.dispatch import complex_gemm_bf16

        return complex_gemm_bf16(ComplexTensor(xr, xi), ComplexTensor(wr, wi))

    # Three cross-terms: hh + hl + lh  (ll is O(u_bf16^4), omitted).
    # Data-dependency trick: add 0 * hh.mean() to the next term's inputs.
    # This creates a graph edge that forces XLA to evaluate hh before starting
    # hl, and hl before starting lh, without requiring any sync() API call.
    # The mean() returns a zero-dim tensor; multiplied by 0 its value is 0,
    # so the numerics are unchanged but the dependency is encoded in the graph.
    # This is more robust than torch_xla.sync() across Neuron SDK versions.
    hh = _term(x_r_h, x_i_h, W_r_h, W_i_h)
    _dep_hh = hh.real.mean() * 0  # zero, but forces hh to be evaluated first

    hl = _term(
        x_r_h + _dep_hh,
        x_i_h + _dep_hh,
        W_r_l,
        W_i_l,
    )
    _dep_hl = hl.real.mean() * 0  # zero, but forces hl to be evaluated first

    lh = _term(
        x_r_l + _dep_hl,
        x_i_l + _dep_hl,
        W_r_h,
        W_i_h,
    )

    X_re = (hh.real + hl.real + lh.real).reshape(orig_shape)
    X_im = (hh.imag + hl.imag + lh.imag).reshape(orig_shape)
    result = ComplexTensor(X_re, X_im)
    if inverse:
        result = result * (1.0 / n)
    return result


def _fft_via_ozaki_hq(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """2-level Ozaki DFT-GEMM: W 3-split × x 3-split → 6 BF16 matmuls, FP32 accumulation.

    Target: ≤ 1e-6 rel error at N ≤ 256. The FP32 output (and FP32 PSUM sum of the
    six terms) sets the floor: measured ~7e-8 (N=64) to ~1.1e-7 (N=256) with exact
    BF16 products (CPU emulation). The earlier "O(u_bf16^4) ≈ 2e-9" claim ignored
    that FP32 floor.

    Split structure (both from the higher-precision source, FP32 residuals or
    better between levels):
      x 3-way: x_h1, x_h2, x_h3 (BF16) from the FP32 input
      W 3-way: W_h1, W_h2, W_h3 (BF16) from the FP64 DFT matrix

    Six cross-terms, triangular (level(W) + level(x) ≤ 2):
      W_h1@x_h1, W_h1@x_h2, W_h1@x_h3, W_h2@x_h1, W_h2@x_h2, W_h3@x_h1

    Before trnfft#81 W was built in FP32 (twiddle error ~N·1e-7) and split only
    2-way (residual ~u_bf16^2), so ozaki_hq landed at ozaki's level (4.9e-6 at
    N=64, 2.0e-5 at N=256 on trn2).

    Cost: 6× BF16 DFT-GEMM ≈ 3.5× FP32 DFT-GEMM.
    Data-dependency trick forces sequential XLA graph execution (same as _fft_via_ozaki).
    """
    n = x.shape[-1]
    orig_shape = x.real.shape

    W_r_fp64, W_i_fp64 = _dft_matrix_fp64(n, inverse)

    x_re_fp32 = x.real.reshape(-1, n).contiguous()
    x_im_fp32 = x.imag.reshape(-1, n).contiguous()

    # 3-way split of W, taken from FP64 (trnfft#81). A 2-way W split leaves a
    # ~u_bf16^2 ≈ 1.5e-5 residual in W itself, capping the result at ~1.3-1.9e-6
    # rel error however finely x is split.
    W_r_h1, W_r_h2, W_r_h3 = _ozaki_split_3way_bf16(W_r_fp64)
    W_i_h1, W_i_h2, W_i_h3 = _ozaki_split_3way_bf16(W_i_fp64)

    # 3-way split of x (FP32 residuals between levels — the 2-level key)
    x_r_h1, x_r_h2, x_r_h3 = _ozaki_split_3way_bf16(x_re_fp32)
    x_i_h1, x_i_h2, x_i_h3 = _ozaki_split_3way_bf16(x_im_fp32)

    def _term(xr, xi, wr, wi) -> ComplexTensor:
        from .nki.dispatch import complex_gemm_bf16

        return complex_gemm_bf16(ComplexTensor(xr, xi), ComplexTensor(wr, wi))

    # Six terms with sequential data-dependencies to force XLA graph execution.
    # Triangular term set: keep W_a@x_b for level(a) + level(b) <= 2 (0-based).
    # Dropped terms are O(u_bf16^3) or smaller.
    t1 = _term(x_r_h1, x_i_h1, W_r_h1, W_i_h1)
    _d1 = t1.real.mean() * 0

    t2 = _term(x_r_h2 + _d1, x_i_h2 + _d1, W_r_h1, W_i_h1)
    _d2 = t2.real.mean() * 0

    t3 = _term(x_r_h3 + _d2, x_i_h3 + _d2, W_r_h1, W_i_h1)
    _d3 = t3.real.mean() * 0

    t4 = _term(x_r_h1 + _d3, x_i_h1 + _d3, W_r_h2, W_i_h2)
    _d4 = t4.real.mean() * 0

    t5 = _term(x_r_h2 + _d4, x_i_h2 + _d4, W_r_h2, W_i_h2)
    _d5 = t5.real.mean() * 0

    t6 = _term(x_r_h1 + _d5, x_i_h1 + _d5, W_r_h3, W_i_h3)

    X_re = (t1.real + t2.real + t3.real + t4.real + t5.real + t6.real).reshape(orig_shape)
    X_im = (t1.imag + t2.imag + t3.imag + t4.imag + t5.imag + t6.imag).reshape(orig_shape)
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
    from .nki.dispatch import _kernel_device, _use_simulator
    from .nki.stockham import stockham_radix4_stage_kernel

    n = x.shape[-1]
    assert _is_power_of_four(n), f"Stockham radix-4 requires N=4^k; got N={n}"

    use_sim = _use_simulator()
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
        padding = torch.zeros(B_pad - B, n, dtype=flat_re.dtype, device=flat_re.device)
        pad_re = torch.cat([flat_re, padding], dim=0)
        pad_im = torch.cat([flat_im, padding], dim=0)

    orig_device = x.real.device
    device = None if use_sim else _kernel_device()

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

        out_re, out_im = _launch_stockham_stage(
            stockham_radix4_stage_kernel,
            re_groups,
            im_groups,
            tw_r,
            tw_i,
            use_sim,
            per_row_cd=True,
            op_name="stockham_radix4_stage",
        )

        # Stockham output permutation: (B, L, M, 4) -> (B, 4, L, M) -> (B, N)
        out_re_4d = out_re.reshape(B_pad, L, M, 4)
        out_im_4d = out_im.reshape(B_pad, L, M, 4)
        re = out_re_4d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)
        im = out_im_4d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)

    if not use_sim:
        re = _to_host(re, orig_device)
        im = _to_host(im, orig_device)
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
    from .nki.dispatch import _kernel_device, _use_simulator
    from .nki.stockham import stockham_radix8_w8_kernel

    n = x.shape[-1]
    assert _is_power_of_eight(n), f"Radix-8 requires N=8^k; got N={n}"

    use_sim = _use_simulator()
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
        padding = torch.zeros(B_pad - B, n, dtype=flat_re.dtype, device=flat_re.device)
        pad_re = torch.cat([flat_re, padding], dim=0)
        pad_im = torch.cat([flat_im, padding], dim=0)

    orig_device = x.real.device
    device = None if use_sim else _kernel_device()

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

        out_re, out_im = _launch_stockham_stage(
            stockham_radix8_w8_kernel,
            pre_re,
            pre_im,
            w8_re,
            w8_im,
            use_sim,
            per_row_cd=False,
            op_name="stockham_radix8_w8",
        )

        # Stockham output permutation: (B, L, M, 8) → (B, 8, L, M) → (B, N)
        out_re_8d = out_re.reshape(B_pad, L, M, 8)
        out_im_8d = out_im.reshape(B_pad, L, M, 8)
        re = out_re_8d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)
        im = out_im_8d.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)

    if not use_sim:
        re = _to_host(re, orig_device)
        im = _to_host(im, orig_device)
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
    from .nki.dispatch import _kernel_device, _use_simulator
    from .nki.stockham import stockham_radix4_stage_kernel, stockham_radix8_w8_kernel

    n = x.shape[-1]
    plan = _mixed_radix_plan(n)

    use_sim = _use_simulator()
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
        padding = torch.zeros(B_pad - B, n, dtype=flat_re.dtype, device=flat_re.device)
        pad_re = torch.cat([flat_re, padding], dim=0)
        pad_im = torch.cat([flat_im, padding], dim=0)

    orig_device = x.real.device
    device = None if use_sim else _kernel_device()

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
            out_re, out_im = _launch_stockham_stage(
                stockham_radix8_w8_kernel,
                pre_re,
                pre_im,
                w8_re,
                w8_im,
                use_sim,
                per_row_cd=False,
                op_name="stockham_radix8_w8",
            )
        else:
            out_re, out_im = _launch_stockham_stage(
                stockham_radix4_stage_kernel,
                re_groups,
                im_groups,
                tw_r,
                tw_i,
                use_sim,
                per_row_cd=True,
                op_name="stockham_radix4_stage",
            )

        out_re_rd = out_re.reshape(B_pad, L, M, r)
        out_im_rd = out_im.reshape(B_pad, L, M, r)
        re = out_re_rd.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)
        im = out_im_rd.permute(0, 3, 1, 2).contiguous().reshape(B_pad, n)
        L *= r

    if not use_sim:
        re = _to_host(re, orig_device)
        im = _to_host(im, orig_device)
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
    instead of the Vector engine (log2(N) butterfly stages); up to
    _FOUR_STEP_MAX the four-step two-GEMM path is used. Both apply to "kahan"
    as well, since they are more accurate than the Kahan butterfly on Trn2
    (trnfft#82). Above _FOUR_STEP_MAX, power-of-2 FP32 "fast" and "kahan"
    take the six-step (from _SIX_STEP_MIN, trnfft#94); with TRNFFT_SIX_STEP=0
    "kahan" uses the compensated butterfly and "fast" uses Stockham. Set
    _DFT_GEMM_THRESHOLD = 0 to force the Kahan butterfly at any N.
    """
    from .nki.butterfly import butterfly_stage_kernel, butterfly_stage_kernel_kahan
    from .nki.dispatch import _kernel_device, _launch, _use_simulator

    kernel = butterfly_stage_kernel_kahan if precision == "kahan" else butterfly_stage_kernel
    op_name = "butterfly_stage_kahan" if precision == "kahan" else "butterfly_stage"
    use_sim = _use_simulator()
    n = x.shape[-1]

    # Bench-toggles: force a specific path ahead of all threshold checks.
    if _FORCE_OZAKI_HQ:
        return _fft_via_ozaki_hq(x, inverse)
    if _FORCE_OZAKI:
        return _fft_via_ozaki(x, inverse)
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

    # Float64 past the GEMM range: every kernel below is FP32-only, and float64
    # operands came back non-finite (#108). That hit a float64 caller at N > 1024
    # (or Bluestein N > 512), and every "double" Bluestein (which promotes to
    # float64) at N > 512. "double" is the accuracy-first mode, so it computes
    # these on the host in FP64. Other modes run the kernels in FP32 and return
    # float64, as the FP32 GEMM paths already do for float64 input.
    if x.real.dtype == torch.float64:
        if precision == "double":
            return _fft_host_fp64(x, inverse)
        y = _cooley_tukey_nki_nograd(
            ComplexTensor(x.real.float(), x.imag.float()), inverse, precision=precision
        )
        return ComplexTensor(y.real.double(), y.imag.double())

    # "bf16" mode: BF16 compute on Tensor Engine, FP32 PSUM → FP32 output.
    if precision == "bf16" and n <= min(_DFT_GEMM_THRESHOLD, _LOWP_GEMM_MAX):
        return _fft_via_gemm_bf16(x, inverse)

    # "bf16_refined" mode: BF16 + one IR step with FP32 residual → ~5e-6 (#86).
    if precision == "bf16_refined" and n <= min(_DFT_GEMM_THRESHOLD, _LOWP_GEMM_MAX):
        return _fft_iterative_refinement(x, inverse, steps=1)

    # "ozaki_hq" mode: 2-level Ozaki, 6 matmuls, O(sqrt(N)*u_bf16^4) error.
    # Requires FP32 product precision in nc_matmul; not available on trn1.
    if precision == "ozaki_hq" and n <= min(_DFT_GEMM_THRESHOLD, _LOWP_GEMM_MAX):
        return _ozaki_or_fallback(x, inverse, hq=True)

    # "ozaki" mode: 1-level BF16 split, 3 matmuls, O(sqrt(N)*u_bf16^2) error.
    # Requires FP32 product precision in nc_matmul; not available on trn1.
    if precision == "ozaki" and n <= min(_DFT_GEMM_THRESHOLD, _LOWP_GEMM_MAX):
        return _ozaki_or_fallback(x, inverse, hq=False)

    # Batch-fused four-step (one launch per batch, trnfft#82 follow-up) from
    # _FOUR_STEP_FUSED_MIN to _FOUR_STEP_MAX; it beats the single GEMM there.
    # A threshold of 0 still disables every GEMM path.
    if _use_fused_four_step(x, n):
        return _fft_via_four_step_fused(x, inverse)

    # "kahan" takes the GEMM paths too (trnfft#82). On Trn2 the single GEMM /
    # four-step measure 1.2e-7-1.6e-7 rel err, below the Kahan butterfly
    # (1.9e-7-4.6e-7, trn1): the PSUM accumulates in FP32 with no per-stage
    # rounding to compensate. Keeping kahan on the butterfly made it *worse*
    # than "fast" once fast moved to the GEMM paths (Bluestein N=997 on Trn2:
    # kahan 4.09e-7 vs fast 2.62e-7). Kahan's compensated chirp multiplies in
    # Bluestein still apply. Threshold 0 still forces the Kahan butterfly.
    if n <= _DFT_GEMM_THRESHOLD:
        return _fft_via_gemm(x, inverse)

    # Four-step FFT as two GEMMs (trnfft#82): 2 kernel launches instead of one
    # per Stockham stage plus inter-stage permutes. A threshold of 0 disables
    # it along with DFT-GEMM ("force butterfly/Stockham").
    if _DFT_GEMM_THRESHOLD > 0 and n <= _FOUR_STEP_MAX and n >= 4 and _is_power_of_two(n):
        return _fft_via_four_step(x, inverse)

    # Six-step (Bailey) large-N FFT (trnfft#54, default since #94): N = N1 * N2 as two
    # batched FFTs of length <= _FOUR_STEP_MAX on the fused four-step kernel,
    # with a twiddle and three transposes on the device.
    if _use_six_step(x, n, precision):
        return _fft_via_six_step(x, inverse, precision)

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
        padding = torch.zeros(B - B_orig, n, dtype=flat_re.dtype, device=flat_re.device)
        pad_re = torch.cat([flat_re, padding], dim=0)
        pad_im = torch.cat([flat_im, padding], dim=0)

    sign = 1.0 if inverse else -1.0

    orig_device = x.real.device
    device = None if use_sim else _kernel_device()

    # Bit-reversal permutation along the last dim.
    indices = _bit_reverse_indices(n, log2n).to(pad_re.device)
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
        # FP64 angles/cos/sin, cast once (trnfft#82; #81 recipe).
        angles = sign * 2.0 * math.pi * torch.arange(half, dtype=torch.float64) / m
        tw_re_1d = torch.cos(angles).to(x.real.dtype)
        tw_im_1d = torch.sin(angles).to(x.real.dtype)
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
            re, im = _launch(kernel, op_name, re, im, tw_re_bcast, tw_im_bcast, n, s)

    if not use_sim:
        re = _to_host(re, orig_device)
        im = _to_host(im, orig_device)
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
    # Every tensor built here lives on the caller's device. The chirp, the
    # zero-padded y and the filter h were created on CPU (torch default), so
    # on a Neuron device ``x * chirp`` raised "input tensor is on cpu device,
    # expected neuron" at every non-power-of-2 length (trnfft#80 follow-up).
    device = x.real.device

    # Chirp sequence: W_N^(k^2/2) = exp(sign * πi * k^2 / N)
    # Built in FP64 with k^2 reduced mod 2N (exp(i*pi*k^2/N) has period 2N in
    # k^2), then cast. The FP32 angle pi*k^2/N reaches ~pi*N, so rounding it
    # gave a chirp error of ~N*1e-7 rel: 1.2e-4 at N=1000 on Trn2 and CPU
    # alike (same bug class as trnfft#81, found in the #82 hardware run).
    k = torch.arange(n, dtype=torch.int64)
    k2 = (k * k) % (2 * n)
    chirp_angles = sign * math.pi * k2.to(torch.float64) / n
    chirp_re = torch.cos(chirp_angles).to(x.dtype)
    chirp_im = torch.sin(chirp_angles).to(x.dtype)

    # Filter h = conj(chirp) with circular wrap, built on CPU in one shot
    # (no per-element device writes), then moved with the chirp.
    h_re = torch.zeros(m, dtype=x.dtype)
    h_im = torch.zeros(m, dtype=x.dtype)
    h_re[:n] = chirp_re
    h_im[:n] = -chirp_im  # conjugate
    if n > 1:
        h_re[m - n + 1 :] = torch.flip(chirp_re[1:], dims=[0])
        h_im[m - n + 1 :] = -torch.flip(chirp_im[1:], dims=[0])  # conjugate

    chirp = ComplexTensor(chirp_re.to(device), chirp_im.to(device))
    h = ComplexTensor(h_re.to(device), h_im.to(device))

    # Step 1: y[n] = x[n] * chirp[n]
    if precision == "kahan":
        y = _complex_mul_kahan(x, chirp)
    else:
        y = x * chirp

    # Step 2: Zero-pad y to length m
    batch_shape = x.shape[:-1]
    y_pad_re = torch.zeros(*batch_shape, m, dtype=x.dtype, device=device)
    y_pad_im = torch.zeros(*batch_shape, m, dtype=x.dtype, device=device)
    y_pad_re[..., :n] = y.real
    y_pad_im[..., :n] = y.imag
    y_padded = ComplexTensor(y_pad_re, y_pad_im)

    # Step 3: filter h (built above, already on ``device``)

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
