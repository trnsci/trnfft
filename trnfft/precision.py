"""Precision-mode selection for trnfft.

Three modes trade off speed vs numerical accuracy:

* ``"fast"`` (default) — straight FP32 throughout. Matches the behavior of
  every trnfft release prior to v0.11.0. Bluestein at N ≥ 500 accumulates
  ~2e-2 relative error from the 3-FFT chain.
* ``"kahan"`` — compensated complex multiply (2Prod + 2Sum) in the Bluestein
  chirp multiplications on the host, plus a Kahan variant of the NKI
  butterfly kernel. Target ~1e-3 rel error at N=8192. Roughly 2× the op
  count of "fast" in the compensated sections.
* ``"double"`` — promotes math to FP64 for maximum accuracy. Affects two paths:

  1. **Bluestein** (non-power-of-2 N): promotes the entire chirp/pad/filter
     pipeline and the three inner FFTs. Target ~1e-6 rel error at any N.

  2. **DFT-GEMM** (power-of-2 N ≤ 1024): bypasses NKI and computes ``W @ x``
     on CPU in FP64. Trainium's PSUM accumulator is always FP32; CPU is the
     only way to get FP64 accumulation. Target ~1e-14 rel error. Slower than
     the "fast" NKI path, but "double" is explicitly the accuracy-first mode.
     For N > 1024, falls through to NKI Stockham (~1e-4 FP32 rel error).

Backend interaction: "kahan" and "double" do not disable the NKI backend
globally — they only change the code path inside ``_bluestein``,
``_cooley_tukey_nki_nograd`` (DFT-GEMM double bypass + "kahan" butterfly
kernel selection), and ``_fft_via_gemm_double``.
"""

from __future__ import annotations

_VALID = ("fast", "kahan", "double")

_precision: str = "fast"


def set_precision(mode: str) -> None:
    """Set the global precision mode. One of {"fast", "kahan", "double"}."""
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
