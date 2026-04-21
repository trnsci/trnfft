"""Stockham radix-4 and radix-8 FFT — CPU reference implementations.

Stockham is an iterative out-of-place radix-r FFT that sidesteps the
bit-reversal permutation by interleaving data reorganization with the
butterfly computation.

Radix-4 (N = 4^k): each stage applies W_4 (adds/subs only) after a
twiddle multiply. The NKI port uses the Vector engine for both ops.

Radix-8 (N = 8^k): each stage applies W_8 (requires real complex
multiplications) after a twiddle multiply. The NKI port uses the Vector
engine for twiddle multiply and the Tensor engine (nc_matmul) for W_8.
Powers of 8: {8, 64, 512, 4096, ...}. N=512 is the primary new coverage
(currently 9 butterfly stages → 3 radix-8 stages).

Mathematical foundation per stage s (radix-r):

    Input:  x of shape (N,), viewed as (L, r, M) where
            L = r^s  (number of already-combined groups)
            M = N / r^(s+1)  (stride within the next combined group)
    Twiddle: T[l, k] = exp(-2πi * l * k / (r * L)) for k ∈ {0..r-1}
    Compute: y[l, j, m] = sum_k (W_r[j, k] * T[l, k] * x[l, k, m])
    Output:  y reshaped and permuted into the next stage's layout
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


# ---------------------------------------------------------------------------
# Radix-8 reference implementation
# ---------------------------------------------------------------------------


def _is_power_of_eight(n: int) -> bool:
    if n <= 0 or n & (n - 1):
        return False
    return int(math.log2(n)) % 3 == 0


def _w8_matvec(a: ComplexTensor) -> ComplexTensor:
    """8-point DFT (W_8) applied along the r=8 axis (dim -2 of 4-D input).

    Input shape: (B, L, 8, M); output same shape.

    W_8[j, k] = exp(-2πi·j·k/8). Unlike W_4, the entries are not ±1/±i
    so a full complex matmul is needed. Because W_8 is symmetric (j·k = k·j)
    we compute y = x @ W_8 (row-vector form, same as y = x @ W_8^T = x @ W_8).
    """
    dtype = a.real.dtype
    idx = torch.arange(8, dtype=dtype)
    ang = -2.0 * math.pi * idx.unsqueeze(1) * idx.unsqueeze(0) / 8  # (8, 8)
    w8_r = torch.cos(ang)  # real part of W_8
    w8_i = torch.sin(ang)  # imag part of W_8

    B, L, _, M = a.real.shape
    # Reshape (B, L, 8, M) → (B*L*M, 8) for batched matmul.
    ar = a.real.permute(0, 1, 3, 2).contiguous().reshape(B * L * M, 8)
    ai = a.imag.permute(0, 1, 3, 2).contiguous().reshape(B * L * M, 8)

    # y_r = a_r @ W_8_r - a_i @ W_8_i
    # y_i = a_r @ W_8_i + a_i @ W_8_r
    yr = ar @ w8_r - ai @ w8_i
    yi = ar @ w8_i + ai @ w8_r

    out_r = yr.reshape(B, L, M, 8).permute(0, 1, 3, 2).contiguous()
    out_i = yi.reshape(B, L, M, 8).permute(0, 1, 3, 2).contiguous()
    return ComplexTensor(out_r, out_i)


def stockham_radix8(x: ComplexTensor, inverse: bool = False) -> ComplexTensor:
    """Iterative Stockham radix-8 FFT along the last dimension.

    Requires ``x.shape[-1]`` to be a power of 8 (8, 64, 512, 4096, ...).
    Accepts arbitrary leading batch dims. CPU/PyTorch reference for the NKI
    port in :mod:`trnfft.nki.stockham`.

    Each stage applies an 8-point DFT (W_8) to groups of 8 elements after
    a twiddle multiply, then performs the Stockham output permutation.
    log_8(N) stages total — fewer than radix-4's log_4(N) for the same N.

    Key difference from radix-4: W_8 entries are exp(-2πi·j·k/8), which
    include ±√2/2 ± i√2/2 and require actual complex multiplications.
    The NKI port routes W_8 through nc_matmul on the Tensor engine.
    """
    n = x.shape[-1]
    assert _is_power_of_eight(n), f"Stockham radix-8 requires N=8^k; got N={n}"

    batch_shape = x.shape[:-1]
    x_re = x.real.reshape(-1, n).contiguous()
    x_im = x.imag.reshape(-1, n).contiguous()
    B = x_re.shape[0]

    if inverse:
        x_im = -x_im

    sign = -1.0
    a = ComplexTensor(x_re.clone(), x_im.clone())
    L = 1  # 8^s: number of already-combined size-8^s sub-FFTs
    while L < n:
        M = n // (8 * L)
        ar = a.real.reshape(B, L, 8, M)
        ai = a.imag.reshape(B, L, 8, M)

        # Twiddle: T[l, k] = exp(sign·2πi·l·k / (8·L)) for k ∈ {0..7}
        l_idx = torch.arange(L, dtype=ar.dtype).view(1, L, 1, 1)
        k_idx = torch.arange(8, dtype=ar.dtype).view(1, 1, 8, 1)
        ang = sign * 2.0 * math.pi * l_idx * k_idx / (8.0 * L)
        tw_r = torch.cos(ang)
        tw_i = torch.sin(ang)
        pre_r = ar * tw_r - ai * tw_i
        pre_i = ar * tw_i + ai * tw_r

        # 8-point DFT along the k-axis.
        mid = _w8_matvec(ComplexTensor(pre_r, pre_i))

        # Stockham output permutation: (B, L, 8, M) → (B, 8, L, M) → (B, N)
        out_r = mid.real.permute(0, 2, 1, 3).contiguous().reshape(B, n)
        out_i = mid.imag.permute(0, 2, 1, 3).contiguous().reshape(B, n)
        a = ComplexTensor(out_r, out_i)
        L *= 8

    result_re = a.real.reshape(*batch_shape, n)
    result_im = a.imag.reshape(*batch_shape, n)
    if inverse:
        result_im = -result_im
        result_re = result_re / n
        result_im = result_im / n
    return ComplexTensor(result_re, result_im)


# ---------------------------------------------------------------------------
# Mixed-radix reference implementation
# ---------------------------------------------------------------------------


def _mixed_radix_plan(n: int) -> list[int]:
    """Optimal [8^a, 4^b] stage sequence for power-of-2 n.

    Minimises total stage count for n = 8^a × 4^b by maximising a.
    All power-of-2 n ≥ 4 can be expressed this way (by the Chicken-McNugget
    theorem for 2 and 3; only n=2 cannot, but n≤256 goes to DFT-GEMM anyway).
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of 2; got {n}")
    k = int(math.log2(n))
    a = k // 3
    while (k - 3 * a) % 2 != 0:
        a -= 1
    b = (k - 3 * a) // 2
    return [8] * a + [4] * b


def stockham_mixed_radix(x: ComplexTensor, inverse: bool = False) -> ComplexTensor:
    """Iterative mixed-radix [8^a, 4^b] Stockham FFT along the last dimension.

    Requires ``x.shape[-1]`` to be a power of 2 ≥ 4. Computes the optimal
    radix decomposition: e.g. N=1024 → [8,8,4,4] (4 stages vs radix-4's 5),
    N=2048 → [8,8,8,4] (4 stages vs butterfly's 11).

    Each stage applies the appropriate DFT matvec (_w8_matvec or _w4_matvec)
    after a twiddle multiply, then the Stockham output permutation.
    """
    n = x.shape[-1]
    plan = _mixed_radix_plan(n)

    batch_shape = x.shape[:-1]
    x_re = x.real.reshape(-1, n).contiguous()
    x_im = x.imag.reshape(-1, n).contiguous()
    B = x_re.shape[0]

    if inverse:
        x_im = -x_im

    sign = -1.0
    a = ComplexTensor(x_re.clone(), x_im.clone())
    L = 1
    for r in plan:
        M = n // (r * L)
        ar = a.real.reshape(B, L, r, M)
        ai = a.imag.reshape(B, L, r, M)

        l_idx = torch.arange(L, dtype=ar.dtype).view(1, L, 1, 1)
        k_idx = torch.arange(r, dtype=ar.dtype).view(1, 1, r, 1)
        ang = sign * 2.0 * math.pi * l_idx * k_idx / (float(r) * L)
        tw_r = torch.cos(ang)
        tw_i = torch.sin(ang)
        pre_r = ar * tw_r - ai * tw_i
        pre_i = ar * tw_i + ai * tw_r

        pre = ComplexTensor(pre_r, pre_i)
        if r == 8:
            mid = _w8_matvec(pre)
        else:
            mid = _w4_matvec(pre)

        out_r = mid.real.permute(0, 2, 1, 3).contiguous().reshape(B, n)
        out_i = mid.imag.permute(0, 2, 1, 3).contiguous().reshape(B, n)
        a = ComplexTensor(out_r, out_i)
        L *= r

    result_re = a.real.reshape(*batch_shape, n)
    result_im = a.imag.reshape(*batch_shape, n)
    if inverse:
        result_im = -result_im
        result_re = result_re / n
        result_im = result_im / n
    return ComplexTensor(result_re, result_im)
