"""NKI kernel: batch-fused four-step (Bailey) FFT, one launch per batch (trnfft#82).

The unfused four-step driver (:func:`trnfft.fft_core._fft_via_four_step`) runs
GEMM(x, W_N1), a twiddle kernel and GEMM(., W_N2) as three launches with three
transpose copies between them. Every one of those passes reads and writes the
whole (B, N) batch in HBM, so at large B the device time grows with B (Trn2,
N=4096: 11.7 ms at B=1024). This kernel does the whole transform for a group
of rows in SBUF/PSUM and touches HBM once on the way in and once on the way out.

Layout: no transposes are needed. Row ``b`` of ``x`` viewed as an
``(N1, N2)`` matrix ``X[n1, n2] = x[b, N2*n1 + n2]`` is already partition-major
in ``n1``, which is the contraction axis of step 1::

    Y[n2, k1]  = sum_n1 X[n1, n2] W1[n1, k1]    nc_matmul(stationary=X,  moving=W1)
    Y[n2, k1] *= T[n2, k1]                      twiddle, Vector engine
    O[k2, k1]  = sum_n2 W2[n2, k2] Y[n2, k1]    nc_matmul(stationary=W2, moving=Y)

``O[k2, k1]`` is ``X[b, k1 + N1*k2]`` in natural order, i.e. row ``b`` of the
output viewed as ``(N2, N1)``.

Filling the 128x128 PE array (each change measured on Trn2, N=4096, B=1024):

- **Row packs.** P = 128 // N2 rows sit side by side in the stationary
  operand, so step 1 produces P rows' Y on all 128 PSUM partitions, the
  twiddle runs on 128 partitions, and step 2 uses a block-diagonal W2
  (P copies of W_N2, K = 128). One strided DMA per component loads a group;
  one stores it. (3.13 -> 1.31 ms against one row per stationary.)
- **Complex step 1 as one real matmul.** X_re and X_im are stacked on the
  partition (contraction) axis, K = 2*N1, against the real block matrix
  ``[[W1r, W1i], [-W1i, W1r]]``, so a single matmul writes
  ``[Y_re | Y_im]`` side by side in PSUM (1.31 -> 0.79 ms against four
  K = N1 matmuls per pack). Needs N1 <= 64.

A group is Q = 512 // N1 packs (one 512-wide PSUM tile per accumulator).

The twiddle uses the ``_complex_mul_kernel`` formula and step 2 accumulates
real*real before imag*(-imag), as in the unfused driver. Step 1 accumulates
its K = 2*N1 terms in one pass rather than two K = N1 passes, so results can
differ from the unfused driver in the last bit; Trn2 relative error vs FP64
is unchanged (see trnfft#82).

Inputs (all FP32 HBM):

- ``x_re, x_im`` : ``(B, N)``, N = N1*N2, B a multiple of P.
- ``w1_blk``     : ``(2*N1, 2*N1)`` real block form of W_N1 (above).
- ``t_re, t_im`` : ``(P*N2, Q*N1)``, ``T[n2, k1] = W_N^(n2*k1)`` tiled P times
  on the partition axis and Q times on the free axis; P and Q are read from
  this shape.
- ``w2_re, w2_im`` : ``(P*N2, P*N2)`` block-diagonal W_N2 (1/N folded in for
  the inverse).

Returns ``(out_re, out_im)``, ``(B, N)``. Constraints: N1 <= 64, P*N2 <= 128,
Q*N1 <= 512.
"""

from __future__ import annotations

from .dispatch import HAS_NKI

# Moving-operand free size cap for nc_matmul, and FP32 PSUM bank width.
FOUR_STEP_FREE_MAX = 512


def four_step_pack_rows(n2: int, batch: int) -> int:
    """Rows per stationary operand P: the largest power of 2 <= 128 // N2 dividing ``batch``.

    P rows side by side fill P*N2 of the 128 PE columns / PSUM partitions.
    The kernel needs ``batch % P == 0``; an odd batch gets P = 1.
    """
    p = max(1, 128 // n2)
    while p > 1 and batch % p:
        p //= 2
    return p


def four_step_group_packs(n1: int, batch: int, pack: int) -> int:
    """Row packs per group Q: as many N1-wide packs as fit one 512-wide PSUM tile."""
    return max(1, min(FOUR_STEP_FREE_MAX // n1, batch // pack))


if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def four_step_fft_kernel(x_re, x_im, w1_blk, t_re, t_im, w2_re, w2_im):
        """Batch-fused four-step FFT; see the module docstring."""
        B, N = x_re.shape
        n1 = w1_blk.shape[0] // 2
        n2 = N // n1
        PN2 = w2_re.shape[0]  # P * N2: block-diagonal W2, P rows per stationary
        P = PN2 // n2
        QW = t_re.shape[1]
        Q = QW // n1  # row packs per group
        assert n1 * n2 == N, f"N={N} not divisible by N1={n1}"
        assert P * n2 == PN2 and PN2 <= 128, f"W2 size {PN2} must be P*N2 <= 128"
        assert t_re.shape[0] == PN2, f"twiddle partitions {t_re.shape[0]} != P*N2={PN2}"
        assert Q * n1 == QW and QW <= 512, f"twiddle width {QW} must be Q*N1 <= 512"
        assert 2 * n1 <= 128, f"N1={n1} must be <= 64"
        assert B % P == 0, f"batch {B} must be a multiple of P={P}"

        out_re = nl.ndarray((B, N), dtype=x_re.dtype, buffer=nl.shared_hbm)
        out_im = nl.ndarray((B, N), dtype=x_re.dtype, buffer=nl.shared_hbm)

        # Rows are processed in packs of P (b = pack*P + r). Input pack view
        # (pack, r, n1, n2); output pack view (pack, (r, k2), k1): partition
        # p = r*N2 + k2 of a pack sits at offset p*N1 from the pack start.
        n_packs = B // P
        x_re_v = x_re.reshape((n_packs, P, n1, n2))
        x_im_v = x_im.reshape((n_packs, P, n1, n2))
        o_re_v = out_re.reshape((n_packs, PN2, n1))
        o_im_v = out_im.reshape((n_packs, PN2, n1))

        # Constants, loaded once per launch.
        # Real block form of W1, (2*N1, 2*N1): [[W1r, W1i], [-W1i, W1r]].
        # With X_re on partitions 0..N1-1 and X_im on N1..2*N1-1 of the
        # stationary, one matmul gives [Y_re | Y_im] = X^T W1.
        w1b = nl.load(w1_blk)
        w2r = nl.load(w2_re)
        w2i = nl.load(w2_im)
        w2n = nl.negative(w2i)
        tr = nl.load(t_re)
        ti = nl.load(t_im)

        def _group(pack0, packs):
            width = packs * n1
            # X_re and X_im of the group stacked on the partition axis:
            # (2*N1, packs, P, N2), one DMA per component.
            xs = nl.ndarray((2 * n1, packs, P, n2), dtype=x_re.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xs[0:n1], src=x_re_v[pack0 : pack0 + packs].permute((2, 0, 1, 3)))
            nisa.dma_copy(
                dst=xs[n1 : 2 * n1], src=x_im_v[pack0 : pack0 + packs].permute((2, 0, 1, 3))
            )

            # PSUM tiles are zeroed and every matmul accumulates, the pattern of
            # _complex_gemm_kernel. Starting groups with accumulate=False on
            # column slices of one PSUM tile lost accumulated terms on Trn2
            # (only the last row of each group was right).
            y = nl.zeros((PN2, packs, 2, n1), dtype=nl.float32, buffer=nl.psum)
            for q in nl.affine_range(packs):
                st = xs[:, q, :, :].reshape((2 * n1, PN2))  # (2*N1, P*N2) stationary
                yq = y[:, q, :, :].reshape((PN2, 2 * n1))
                # Step 1: [Y_re | Y_im] = X^T W1 for P rows, K = 2*N1.
                nisa.nc_matmul(dst=yq, stationary=st, moving=w1b, accumulate=True)
            y_r = y[:, :, 0, :]
            y_i = y[:, :, 1, :]

            # Twiddle: A = T * Y (same formula as _complex_mul_kernel(T, Y)).
            t_r = tr[:, 0:width].reshape((PN2, packs, n1))
            t_i = ti[:, 0:width].reshape((PN2, packs, n1))
            ar3 = nl.subtract(nl.multiply(t_r, y_r), nl.multiply(t_i, y_i))
            ai3 = nl.add(nl.multiply(t_r, y_i), nl.multiply(t_i, y_r))
            ar = ar3.reshape((PN2, width))
            ai = ai3.reshape((PN2, width))

            # Step 2: O = blockdiag(W2)^T A (contraction over the P*N2 partitions).
            o_r = nl.zeros((PN2, width), dtype=nl.float32, buffer=nl.psum)
            o_i = nl.zeros((PN2, width), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=o_r, stationary=w2r, moving=ar, accumulate=True)
            nisa.nc_matmul(dst=o_r, stationary=w2n, moving=ai, accumulate=True)
            nisa.nc_matmul(dst=o_i, stationary=w2r, moving=ai, accumulate=True)
            nisa.nc_matmul(dst=o_i, stationary=w2i, moving=ar, accumulate=True)

            or_sb = nl.ndarray((PN2, packs, n1), dtype=nl.float32, buffer=nl.sbuf)
            oi_sb = nl.ndarray((PN2, packs, n1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=or_sb.reshape((PN2, width)), src=o_r)
            nisa.tensor_copy(dst=oi_sb.reshape((PN2, width)), src=o_i)
            nl.store(o_re_v[pack0 : pack0 + packs].permute((1, 0, 2)), value=or_sb)
            nl.store(o_im_v[pack0 : pack0 + packs].permute((1, 0, 2)), value=oi_sb)

        n_full = n_packs // Q
        for g in nl.affine_range(n_full):
            _group(g * Q, Q)
        tail = n_packs - n_full * Q
        if tail > 0:
            _group(n_full * Q, tail)

        return out_re, out_im
