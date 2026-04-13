"""Precision-mode selection for trnfft.

Three modes trade off speed vs numerical accuracy:

* ``"fast"`` (default) — straight FP32 throughout. Matches the behavior of
  every trnfft release prior to v0.11.0. Bluestein at N ≥ 500 accumulates
  ~2e-2 relative error from the 3-FFT chain.
* ``"kahan"`` — compensated complex multiply (2Prod + 2Sum) in the Bluestein
  chirp multiplications on the host, plus a Kahan variant of the NKI
  butterfly kernel. Target ~1e-3 rel error at N=8192. Roughly 2× the op
  count of "fast" in the compensated sections.
* ``"double"`` — promotes the Bluestein host math to FP64 for the entire
  chirp/pad/filter pipeline and the three inner FFTs. Casts result back
  to input dtype. Target ~1e-6 rel error at any N. Only affects Bluestein;
  power-of-2 FFTs are unchanged (they don't have the chain-error issue).

Backend interaction: "kahan" and "double" do not disable the NKI backend
globally — they only change the code path inside ``_bluestein`` and (for
"kahan") the choice of butterfly kernel in ``_cooley_tukey_nki``.
"""

from __future__ import annotations

_VALID = ("fast", "kahan", "double")

_precision: str = "fast"


def set_precision(mode: str) -> None:
    """Set the global precision mode. One of {"fast", "kahan", "double"}."""
    global _precision
    if mode not in _VALID:
        raise ValueError(
            f"precision mode must be one of {_VALID}; got {mode!r}"
        )
    _precision = mode


def get_precision() -> str:
    """Return the current global precision mode."""
    return _precision


def _resolve(precision) -> str:
    """Resolve an optional per-call precision override against the global."""
    if precision is None:
        return _precision
    if precision not in _VALID:
        raise ValueError(
            f"precision must be one of {_VALID} or None; got {precision!r}"
        )
    return precision
