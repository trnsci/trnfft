"""Stockham radix-4 FFT — CPU reference implementation.

Stockham is an iterative out-of-place radix-r FFT that sidesteps the
bit-reversal permutation by interleaving data reorganization with the
butterfly computation. For power-of-4 N:

    N = 4^k
    x' = apply log_4(N) stages, each one a 4×4 DFT applied to groups
         of 4 elements with a fixed Stockham permutation on output.

This CPU reference exists to lock in the indexing + twiddle math for the
later NKI port (see trnfft/nki/stockham.py). The reference is a
straightforward port of the Glassman/Singleton Stockham formulation
adapted for radix-4. It runs on plain PyTorch tensors (any device)
and is independently testable against numpy.fft.

Mathematical foundation per stage s (working in complex tensors):

    Input:  x of shape (N,), viewed as (L, 4, M) where
            L = 4^s  (number of already-combined groups)
            M = N / 4^(s+1)  (stride within the next combined group)
    Twiddle: T[l, m] = exp(-2πi * l * m / (L * 4)) for out-index computation
    Compute: y[l, r, m] = sum_k (W4[r, k] * T[l, k]_mix * x[l, k, m])
    Output:  y reshaped and permuted into the next stage's layout

The 4×4 DFT matrix W_4 has only {1, i, -1, -i} coefficients — no actual
multiplications needed for W_4 itself, just sign flips and real/imag
swaps. The per-stage cost is thus dominated by the twiddle multiply.
That matters on Trainium because the NKI port can implement W_4's
matvec via reorderings + adds, keeping the Tensor engine free for the
twiddle multiply.
"""

from __future__ import annotations

import math

import torch

from .complex import ComplexTensor


def _is_power_of_four(n: int) -> bool:
    if n <= 0:
        return False
    if n & (n - 1):
        return False  # not a power of 2
    # log2(n) must be even for n to be a power of 4.
    return (int(math.log2(n)) & 1) == 0


def _w4_matvec(a: ComplexTensor) -> ComplexTensor:
    """4-point DFT (W_4) applied along the leading r=4 axis.

    Input shape: (..., 4, ...); output same shape. The W_4 DFT matrix is:
        [ 1   1   1   1]
        [ 1  -i  -1   i]
        [ 1  -1   1  -1]
        [ 1   i  -1  -i]

    So: y[0] = a[0] +  a[1] +  a[2] +  a[3]
        y[1] = a[0] - ia[1] -  a[2] + ia[3]
        y[2] = a[0] -  a[1] +  a[2] -  a[3]
        y[3] = a[0] + ia[1] -  a[2] - ia[3]

    Multiply by i means (re, im) -> (-im, re); by -i means (re, im) -> (im, -re).
    No scalar multiplies needed — just adds, subtracts, and swaps.
    Using the inline expressions here keeps the CPU reference faithful
    to what the NKI port will execute (which also avoids literal ±i
    multiplication for the same reason).
    """
    a0r, a0i = a.real[..., 0, :], a.imag[..., 0, :]
    a1r, a1i = a.real[..., 1, :], a.imag[..., 1, :]
    a2r, a2i = a.real[..., 2, :], a.imag[..., 2, :]
    a3r, a3i = a.real[..., 3, :], a.imag[..., 3, :]

    # y[0] = a0 + a1 + a2 + a3
    y0r = a0r + a1r + a2r + a3r
    y0i = a0i + a1i + a2i + a3i
    # y[1] = a0 - ia1 - a2 + ia3
    #   -i*a1 = (a1i, -a1r);  i*a3 = (-a3i, a3r)
    y1r = a0r + a1i - a2r - a3i
    y1i = a0i - a1r - a2i + a3r
    # y[2] = a0 - a1 + a2 - a3
    y2r = a0r - a1r + a2r - a3r
    y2i = a0i - a1i + a2i - a3i
    # y[3] = a0 + ia1 - a2 - ia3
    #   i*a1 = (-a1i, a1r);  -i*a3 = (a3i, -a3r)
    y3r = a0r - a1i - a2r + a3i
    y3i = a0i + a1r - a2i - a3r

    out_r = torch.stack([y0r, y1r, y2r, y3r], dim=-2)
    out_i = torch.stack([y0i, y1i, y2i, y3i], dim=-2)
    return ComplexTensor(out_r, out_i)


def stockham_radix4(x: ComplexTensor, inverse: bool = False) -> ComplexTensor:
    """Iterative Stockham radix-4 FFT along the last dimension.

    Requires ``x.shape[-1]`` to be a power of 4. Accepts arbitrary leading
    batch dims (flattened internally). This is the pure-PyTorch reference;
    the NKI port (:mod:`trnfft.nki.stockham`) mirrors the same per-stage
    structure but swaps the per-group 4-point DFT for a Tensor-engine
    matmul.

    The Stockham algorithm interleaves data reordering with computation:
    stage s reads from buffer A in the current layout and writes to
    buffer B in the next stage's layout. After log_4(N) stages the
    output is in natural index order — no bit-reversal permutation
    required.

    Twiddle factors at stage s for the k-th output of an L=4^s group:
        W[l, k] = exp(sign * 2πi * l * k / (4 * L))  for k in {0,1,2,3}
    where sign = -1 (forward) or +1 (inverse).
    """
    n = x.shape[-1]
    assert _is_power_of_four(n), f"Stockham radix-4 requires N=4^k; got N={n}"

    batch_shape = x.shape[:-1]
    x_re = x.real.reshape(-1, n).contiguous()
    x_im = x.imag.reshape(-1, n).contiguous()
    B = x_re.shape[0]

    # Inverse FFT via the conjugate trick: ifft(X) = (1/N) * conj(fft(conj(X))).
    # Keeps the forward W_4 matvec and positive-sign twiddle chain as the
    # single authoritative path — avoids having two mirror-image kernels and
    # the risk of their permutations falling out of sync.
    if inverse:
        x_im = -x_im

    sign = -1.0

    a = ComplexTensor(x_re.clone(), x_im.clone())
    L = 1  # 4^s: number of already-combined size-4^s sub-FFTs
    while L < n:
        # Reshape (B, N) -> (B, L, 4, M) where M = N / (4L).
        M = n // (4 * L)
        ar = a.real.reshape(B, L, 4, M)
        ai = a.imag.reshape(B, L, 4, M)

        # Apply twiddles: T[l, k, m] = exp(sign*2πi*l*k / (4L))
        #   k dim only — broadcast over l and m.
        # Pre-twiddle:  x'[l, k, m] = x[l, k, m] * exp(sign*2πi*l*k/(4L))
        l_idx = torch.arange(L, dtype=ar.dtype).view(1, L, 1, 1)
        k_idx = torch.arange(4, dtype=ar.dtype).view(1, 1, 4, 1)
        ang = sign * 2.0 * math.pi * l_idx * k_idx / (4.0 * L)
        tw_r = torch.cos(ang)
        tw_i = torch.sin(ang)
        pre_r = ar * tw_r - ai * tw_i
        pre_i = ar * tw_i + ai * tw_r

        # 4-point DFT along the k-axis (axis=-2 of the (B, L, 4, M) tensor).
        mid = _w4_matvec(ComplexTensor(pre_r, pre_i))

        # Stockham output permutation: stride-4 scatter.
        # mid has shape (B, L, 4, M); rearrange to (B, 4, L, M) then
        # reshape to (B, N) with the combined groups now of size 4L.
        out_r = mid.real.permute(0, 2, 1, 3).contiguous().reshape(B, n)
        out_i = mid.imag.permute(0, 2, 1, 3).contiguous().reshape(B, n)
        a = ComplexTensor(out_r, out_i)

        L *= 4

    result_re = a.real.reshape(*batch_shape, n)
    result_im = a.imag.reshape(*batch_shape, n)
    if inverse:
        # Undo the input conjugation; scale by 1/N for the inverse.
        result_im = -result_im
        result_re = result_re / n
        result_im = result_im / n
    return ComplexTensor(result_re, result_im)
