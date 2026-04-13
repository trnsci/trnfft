"""
FFT plan: precomputed twiddle factors and algorithm selection.

Like FFTW/cuFFT, create a plan once, execute many times.
Plans are cached by (size, dtype, inverse).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class FFTAlgorithm(Enum):
    COOLEY_TUKEY = auto()
    BLUESTEIN = auto()


@dataclass(frozen=True)
class FFTPlan:
    n: int
    algorithm: FFTAlgorithm
    inverse: bool
    padded_n: int | None = None  # For Bluestein

    @property
    def is_power_of_2(self) -> bool:
        return self.algorithm == FFTAlgorithm.COOLEY_TUKEY


_plan_cache: dict[tuple, FFTPlan] = {}


def create_plan(n: int, inverse: bool = False) -> FFTPlan:
    key = (n, inverse)
    if key in _plan_cache:
        return _plan_cache[key]

    if n & (n - 1) == 0:
        plan = FFTPlan(n=n, algorithm=FFTAlgorithm.COOLEY_TUKEY, inverse=inverse)
    else:
        m = 1 << (2 * n - 2).bit_length()
        plan = FFTPlan(n=n, algorithm=FFTAlgorithm.BLUESTEIN, inverse=inverse, padded_n=m)

    _plan_cache[key] = plan
    return plan


def clear_plan_cache():
    _plan_cache.clear()
