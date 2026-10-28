"""Precision-mode selection for trnfft.

Three modes trade off speed vs numerical accuracy:

* ``"fast"`` (default) — straight FP32 throughout. Matches the behavior of
  every trnfft release prior to v0.11.0. Bluestein (non-power-of-2 N) measures
  ~4e-7 rel error at N=1000/4097 on CPU since the chirp is built in FP64 with
  k² reduced mod 2N (trnfft#82). Before that the FP32 chirp angle gave 1.2e-4
  at N=1000 and 4.4e-4 at N=4097.
* ``"kahan"`` — compensated complex multiply (Dekker 2Prod) in the Bluestein
  chirp multiplications on the host, plus a Kahan variant of the NKI
  butterfly kernel. Measured butterfly rel error (trn1, SDK 2.29, 2026-04-22):

  .. code-block:: text

     N=256:  fast=1.41e-6  kahan=1.92e-7  (7.3× improvement)
     N=512:  fast=2.15e-6  kahan=2.69e-7  (8.0× improvement)
     N=1024: fast=2.04e-6  kahan=3.02e-7  (6.8× improvement)
     N=4096: fast=3.60e-6  kahan=4.55e-7  (7.9× improvement)

  Roughly 2× the op count of "fast" in the compensated sections. Use when
  forward-error budget for butterfly stages is ≤ 3e-7 (vs "fast"'s ≤ 4e-6).

  Since trnfft#82, power-of-2 FFTs with N ≤ 4096 on the NKI path use the same
  single-GEMM / four-step GEMM paths as "fast" in both modes. On Trn2 those
  measure 1.2e-7 to 1.6e-7, which beats the Kahan butterfly. The Kahan
  butterfly applies above N=4096, or at any N with ``_DFT_GEMM_THRESHOLD = 0``.
  Bluestein keeps its compensated chirp multiplies. The contract is that
  "kahan" is never less accurate than "fast".
* ``"double"`` — promotes math to FP64 for maximum accuracy. Affects two paths:

  1. **Bluestein** (non-power-of-2 N): promotes the entire chirp/pad/filter
     pipeline and the three inner FFTs. Target ~1e-6 rel error at any N.

  2. **DFT-GEMM** (power-of-2 N ≤ 1024): bypasses NKI and computes ``W @ x``
     on CPU in FP64. Trainium's PSUM accumulator is always FP32; CPU is the
     only way to get FP64 accumulation. Target ~1e-14 rel error. Slower than
     the "fast" NKI path, but "double" is explicitly the accuracy-first mode.
     For N > 1024, falls through to NKI Stockham (~1e-4 FP32 rel error).

* ``"bf16"`` — BF16 DFT-GEMM for N ≤ 256 on NKI hardware. W and x computed
  in BF16 on the Tensor Engine; the FP32 PSUM accumulator is preserved as
  output rather than rounded back to BF16 ("PSUM is a free FP32 accumulator").
  Rounding W and x to BF16 sets the error: normwise relative error about
  u_bf16 = 2^-9, growing slowly with log N. CPU emulation gives 1.9e-3 (N=2) to
  2.16e-3 (N=256), max over 20 seeds. Trn2 measured 2.06e-3 / 2.09e-3 at
  N=128/256 (SDK 2.33, trnfft#86), which is the CPU value to 3 digits.
  **Bound: 3e-3** for N ≤ 256. The earlier "≈1e-3 at N=256" figure was
  optimistic, and a 2e-3 reading of it left no headroom over u_bf16.

* ``"bf16_refined"`` — BF16 DFT-GEMM + one iterative-refinement step (IR-1).
  Algorithm: X̂ = fft_bf16(x); r = x − ifft_fp32(X̂); X̂ += fft_bf16(r). The
  reconstruction ifft_fp32 is the FP32 DFT-GEMM with an exact (FP64-built) W.
  That is what cancels the O(u_bf16) error of the BF16 W. Through trnfft#86 the
  reconstruction also used BF16, so the step corrected nothing. Trn2 measured
  1.95e-3 / 2.03e-3 / 2.08e-3 at N=64/128/256, the same as "bf16". With the
  FP32 reconstruction, CPU emulation gives 3.6e-6 (N=2) to 4.9e-6 (N=256),
  forward and inverse. **Bound: 1e-5** for N ≤ 256. Cost: two BF16 GEMMs plus
  one FP32 GEMM, which is more than "fast" (one FP32 GEMM, ~1.5e-7 to 3e-7).
  "bf16_refined" is not a faster route to FP32 accuracy. Above N=256 it falls
  through to the "fast" paths.

* ``"ozaki_hq"`` — 2-level Ozaki (3-way x split + 3-way W split, triangular
  6-term product) for N ≤ 256. 6 BF16 matmuls. Target ≤ 1e-6 rel error; the FP32
  output floor gives ~7e-8 (N=64) to ~1.1e-7 (N=256) with exact BF16 products
  (CPU emulation, after trnfft#81). The earlier "O(u_bf16^4) ≈ 2e-9" figure ignored
  the FP32 floor. Before #81 (FP32 twiddles, 2-way W split) trn2 measured
  4.9e-6 / 9.9e-6 / 2.0e-5 at N=64/128/256, the same as "ozaki" (spike B5, SDK 2.33).
  **Hardware-gated**: requires nc_matmul to compute BF16×BF16 products at FP32
  precision before PSUM accumulation. trn1 measured ozaki_hq ≈ bf16 ≈ 1.7e-3
  (SDK 2.29.0). Emits RuntimeWarning and falls back to "bf16" unless
  ``trnfft.set_ozaki_product_precision_verified(True)`` has been called.

* ``"ozaki"`` — 1-level Ozaki-scheme DFT-GEMM for N ≤ 256 on NKI hardware.
  3 BF16 matmuls (W_h@x_h + W_h@x_l + W_l@x_h). Bound 1.6e-5 rel error at
  N ≤ 256 (O(u_bf16^2)); CPU emulation gives ~3.5e-6–4e-6 after trnfft#81. Before
  #81, trn2 measured 5.9e-6 / 1.0e-5 / 2.0e-5 at N=64/128/256, the N=256 value over
  the bound, all of it from FP32 twiddle generation. **Hardware-gated**: same
  product-precision constraint as "ozaki_hq". On trn1, measured error equals
  single-pass BF16. Emits RuntimeWarning and falls back to "bf16" unless verified.
  Run TestOzakiHQCharacterization to test your hardware, then call
  set_ozaki_product_precision_verified(True) to enable.

  Both Ozaki modes dispatch only for N ≤ 256 (``_LOWP_GEMM_MAX``). At larger
  N, "ozaki"/"ozaki_hq" fall through to the FP32 single-GEMM (N ≤ 1024) or
  four-step (N ≤ 4096) path, which has the "fast" accuracy (~1.5e-7 on Trn2)
  with no Ozaki correction.


Backend interaction: "kahan" and "double" do not disable the NKI backend
globally — they only change the code path inside ``_bluestein``,
``_cooley_tukey_nki_nograd`` (DFT-GEMM double bypass + "kahan" butterfly
kernel selection), and ``_fft_via_gemm_double``. The "bf16" and "bf16_refined"
modes route through ``_fft_via_gemm_bf16`` and ``_fft_iterative_refinement``
respectively, which call ``complex_gemm_bf16`` (the PSUM-FP32 kernel).
``_fft_iterative_refinement`` also calls ``complex_gemm`` (FP32) for the
residual reconstruction.
"""

from __future__ import annotations

_VALID = ("fast", "kahan", "double", "bf16", "bf16_refined", "ozaki", "ozaki_hq")

_precision: str = "fast"


def set_precision(mode: str) -> None:
    """Set the global precision mode. One of ``_VALID``."""
    global _precision
    if mode not in _VALID:
        raise ValueError(f"precision mode must be one of {_VALID}; got {mode!r}")
    _precision = mode


def get_precision() -> str:
    """Return the current global precision mode."""
    return _precision


def _resolve(precision) -> str:
    """Resolve an optional per-call precision override against the global."""
    if precision is None:
        return _precision
    if precision not in _VALID:
        raise ValueError(f"precision must be one of {_VALID} or None; got {precision!r}")
    return precision
