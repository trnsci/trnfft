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


def fft_core(x: ComplexTensor, inverse: bool = False, plan: Optional[FFTPlan] = None) -> ComplexTensor:
    """Compute 1-D FFT along last dimension.

    If plan is None, one is created (and cached) automatically.
    """
    n = x.shape[-1]
    if n == 1:
        return x.clone()

    if plan is None:
        plan = create_plan(n, inverse=inverse)

    if plan.algorithm == FFTAlgorithm.COOLEY_TUKEY:
        return _cooley_tukey(x, inverse)
    else:
        return _bluestein(x, inverse, padded_n=plan.padded_n)


def _cooley_tukey(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Iterative radix-2 decimation-in-time FFT.

    Standard textbook algorithm:
    1. Bit-reversal permutation
    2. log2(N) butterfly stages
    3. For inverse: divide by N at the end
    """
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


def _bluestein(x: ComplexTensor, inverse: bool, padded_n: Optional[int] = None) -> ComplexTensor:
    """Bluestein's algorithm (chirp-z) for arbitrary-size FFT.

    Converts length-N DFT into circular convolution of length M >= 2N-1
    (M is next power of 2), computed via three power-of-2 FFTs.
    """
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
    Y = _cooley_tukey(y_padded, inverse=False)
    H = _cooley_tukey(h.unsqueeze(0) if len(batch_shape) > 0 else h, inverse=False)
    product = Y * H
    conv = _cooley_tukey(product, inverse=True)

    # Step 5: Multiply by chirp and take first N elements
    result_re = conv.real[..., :n]
    result_im = conv.imag[..., :n]
    result = ComplexTensor(result_re, result_im) * chirp

    if inverse:
        result = result * (1.0 / n)

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
