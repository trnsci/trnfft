#!/usr/bin/env python
"""Accuracy table of the double-word FFT (trnfft#172), on CPU.

Correct bits = -log2(||X - X_ref||_2 / ||X_ref||_2) per transform, the worst
over the batch rows, against an mpmath FFT at 200 bits. Inputs are random
complex Gaussians rounded to double-word-exact float64 values (hi + lo of the
split), so every arm sees exactly the same numbers and the reference is the
exact DFT of them. Arms:

  dw-accurate / dw-sloppy  trnfft.dw reference (= the kernel, bit for bit)
  host fp64                torch.fft.fft on complex128 (pocketfft)
  fp32 torch               torch.fft.fft on complex64
  fp32 trnfft              trnfft "fast" FP32 Cooley-Tukey (pytorch backend)

Usage: python scripts/dw_accuracy.py [--rows 4] [--sizes 4,8,...,4096] [--inverse]
"""

from __future__ import annotations

import argparse
import math

import mpmath
import numpy as np
import torch

import trnfft
from trnfft import dw
from trnfft.complex import ComplexTensor


def mp_fft(x, inverse=False, prec=200):
    """Iterative radix-2 FFT in mpmath (unnormalised), x a list of complex."""
    with mpmath.workprec(prec):
        n = len(x)
        m = n.bit_length() - 1
        a = [mpmath.mpc(complex(v).real, complex(v).imag) for v in x]
        rev = [int(bin(i)[2:].zfill(m)[::-1], 2) if m else 0 for i in range(n)]
        a = [a[r] for r in rev]
        sgn = 1 if inverse else -1
        L = 2
        while L <= n:
            h = L // 2
            ws = [mpmath.expjpi(sgn * mpmath.mpf(2 * k) / L) for k in range(h)]
            for s in range(0, n, L):
                for k in range(h):
                    u, v = a[s + k], a[s + k + h] * ws[k]
                    a[s + k], a[s + k + h] = u + v, u - v
            L *= 2
        return a


def bits(got: np.ndarray, ref) -> float:
    """Normwise correct bits of one transform against the mpmath result."""
    with mpmath.workprec(200):
        num = mpmath.fsum(abs(mpmath.mpc(complex(g)) - r) ** 2 for g, r in zip(got, ref))
        den = mpmath.fsum(abs(r) ** 2 for r in ref)
        if num == 0:
            return 53.0
        return float(-mpmath.log(mpmath.sqrt(num / den), 2))


def dw_exact_input(rows, n, rng):
    x = rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))
    d = dw.to_dw(torch.from_numpy(x))
    return d.to_complex128().numpy(), d  # hi + lo is exact in float64


def arms(x: np.ndarray, d, inverse: bool):
    xt = torch.from_numpy(x)
    out = {}
    for add in dw.ADD_MODES:
        y = dw.fft_dw_raw(d, inverse, add=add, backend="reference").to_complex128().numpy()
        out[f"dw-{add}"] = y
    f = torch.fft.ifft if inverse else torch.fft.fft
    n = x.shape[-1]
    scale = n if inverse else 1  # compare unnormalised transforms
    out["host fp64"] = f(xt).numpy() * scale
    out["fp32 torch"] = f(xt.to(torch.complex64)).numpy().astype(np.complex128) * scale
    trnfft.set_backend("pytorch")
    ct = ComplexTensor(xt.real.float().contiguous(), xt.imag.float().contiguous())
    y = trnfft.fft_core.fft_core(ct, inverse=inverse, precision="fast")
    out["fp32 trnfft"] = (y.real.double() + 1j * y.imag.double()).numpy() * scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--sizes", default=",".join(str(1 << m) for m in range(2, 13)))
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--seed", type=int, default=172)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    names = None
    for n in [int(s) for s in a.sizes.split(",")]:
        x, d = dw_exact_input(a.rows, n, rng)
        refs = [mp_fft(list(row), a.inverse) for row in x]
        res = arms(x, d, a.inverse)
        if names is None:
            names = list(res)
            print("| N | " + " | ".join(names) + " |")
            print("|---" * (len(names) + 1) + "|")
        worst = {k: min(bits(res[k][i], refs[i]) for i in range(a.rows)) for k in names}
        print(f"| {n} | " + " | ".join(f"{worst[k]:.1f}" for k in names) + " |", flush=True)
    print(f"\nbits = worst of {a.rows} rows, normwise; u_dw = 2^-48 ({math.log2(2**48):.0f} bits)")


if __name__ == "__main__":
    main()
