"""
NKI butterfly kernel for radix-2 Cooley-Tukey FFT on Trainium.

Accepts batched input (B, n) and vectorizes across B within a single kernel
call per stage. Prior versions looped over batches in Python, which was
catastrophic for multi-call paths (fft2, fftn, batched 1D FFT, STFT) —
each row paid full XLA dispatch overhead.

Layout
------
Input (B, n) is reshaped to (B * num_groups, m) where
  m          = 2 ** (stage + 1)         -- butterfly group size at this stage
  num_groups = n // m                   -- number of groups per batch row
  B          = batch size (arbitrary)

The partition dim becomes `total_groups = B * num_groups`, so we vectorize
over both batches AND butterfly groups simultaneously.

NKI 2.24 constraints:
- Partition dim must be the first dim of any SBUF tile and ≤ 128
- For power-of-2 B, n, stage, total_groups is always a power of 2, so we tile
  the partition dim in constant chunks of min(total_groups, 128)

Twiddle factors are broadcast on the host to shape (total_groups, half) —
the same twiddle values repeated across each batch row — because NKI 2.24
element-wise ops require matching partition dims.

Validated against neuronxcc 2.24.5133.0 on Deep Learning AMI Neuron PyTorch
2.9 (Ubuntu 24.04).
"""

from __future__ import annotations

from .dispatch import HAS_NKI

if HAS_NKI:
    import nki
    import nki.language as nl

    PMAX = 128

    @nki.jit
    def butterfly_stage_kernel(x_re, x_im, tw_re_bcast, tw_im_bcast, n: int, stage: int):
        """Batched radix-2 butterfly stage.

        Parameters
        ----------
        x_re, x_im                 : (B, n) input real/imag in HBM
        tw_re_bcast, tw_im_bcast   : (B * num_groups, half) twiddle factors, pre-broadcast
                                     by the host so every partition row has the right
                                     per-butterfly-position twiddle.
        n                          : transform size (power of 2)
        stage                      : stage index in [0, log2(n))

        Returns
        -------
        out_re, out_im : (B, n) output real/imag in HBM
        """
        B, _ = x_re.shape
        m = 1 << (stage + 1)
        half = m >> 1
        num_groups = n // m
        total_groups = B * num_groups

        out_re = nl.ndarray((B, n), dtype=x_re.dtype, buffer=nl.shared_hbm)
        out_im = nl.ndarray((B, n), dtype=x_im.dtype, buffer=nl.shared_hbm)

        # Flatten (B, n) -> (total_groups, m). Each row is one independent
        # butterfly group. Partition dim = total_groups.
        x_re_2d = x_re.reshape((total_groups, m))
        x_im_2d = x_im.reshape((total_groups, m))
        out_re_2d = out_re.reshape((total_groups, m))
        out_im_2d = out_im.reshape((total_groups, m))

        # NKI affine_range can't evaluate min() symbolically, so use a constant
        # chunk size. For power-of-2 B, n, stage, total_groups is always a power
        # of 2 and PMAX = 128, so the division is exact and no tail tile is needed.
        groups_chunk = total_groups if total_groups <= PMAX else PMAX
        assert total_groups % groups_chunk == 0, (
            f"total_groups={total_groups} (B={B}, num_groups={num_groups}) "
            f"not divisible by chunk size {groups_chunk}"
        )
        n_partition_tiles = total_groups // groups_chunk

        for p in nl.affine_range(n_partition_tiles):
            p_off = p * groups_chunk
            p_end = p_off + groups_chunk  # constant slice size

            # Process each butterfly position k within this partition tile.
            for k in nl.affine_range(half):
                # Twiddle for this column, broadcast to match partition dim.
                t_re_col = nl.load(tw_re_bcast[p_off:p_end, k : k + 1])
                t_im_col = nl.load(tw_im_bcast[p_off:p_end, k : k + 1])

                # Even and odd columns.
                e_re = nl.load(x_re_2d[p_off:p_end, k : k + 1])
                e_im = nl.load(x_im_2d[p_off:p_end, k : k + 1])
                o_re = nl.load(x_re_2d[p_off:p_end, k + half : k + half + 1])
                o_im = nl.load(x_im_2d[p_off:p_end, k + half : k + half + 1])

                # Complex multiply: (t_re + i*t_im) * (o_re + i*o_im)
                # NKI 0.3.0: use nl.multiply / nl.add / nl.subtract explicitly;
                # Python arithmetic operators on NkiTensor are no longer defined.
                prod_re = nl.subtract(nl.multiply(t_re_col, o_re), nl.multiply(t_im_col, o_im))
                prod_im = nl.add(nl.multiply(t_re_col, o_im), nl.multiply(t_im_col, o_re))

                # Butterfly: even = e + prod, odd = e - prod
                nl.store(out_re_2d[p_off:p_end, k : k + 1], value=nl.add(e_re, prod_re))
                nl.store(out_im_2d[p_off:p_end, k : k + 1], value=nl.add(e_im, prod_im))
                nl.store(
                    out_re_2d[p_off:p_end, k + half : k + half + 1],
                    value=nl.subtract(e_re, prod_re),
                )
                nl.store(
                    out_im_2d[p_off:p_end, k + half : k + half + 1],
                    value=nl.subtract(e_im, prod_im),
                )

        return out_re, out_im

    @nki.jit
    def butterfly_stage_kernel_kahan(x_re, x_im, tw_re_bcast, tw_im_bcast, n: int, stage: int):
        """Kahan-compensated variant of ``butterfly_stage_kernel``.

        Uses Dekker 2Prod to split each ``t * o`` product into (hi, lo) and
        accumulates the complex multiply as

            prod_re = (hi_rr - hi_ii) + (lo_rr - lo_ii)
            prod_im = (hi_ri + hi_ir) + (lo_ri + lo_ir)

        The ``lo_*`` terms recover the mantissa bits that are ordinarily
        lost when ``t_re*o_re`` and ``t_im*o_im`` are close in magnitude.
        Roughly 2× the op count of the stock kernel; opt-in via
        ``set_precision("kahan")``.

        Dekker split constant for FP32 is ``2^12 + 1 = 4097``.
        """
        B, _ = x_re.shape
        m = 1 << (stage + 1)
        half = m >> 1
        num_groups = n // m
        total_groups = B * num_groups

        out_re = nl.ndarray((B, n), dtype=x_re.dtype, buffer=nl.shared_hbm)
        out_im = nl.ndarray((B, n), dtype=x_im.dtype, buffer=nl.shared_hbm)

        x_re_2d = x_re.reshape((total_groups, m))
        x_im_2d = x_im.reshape((total_groups, m))
        out_re_2d = out_re.reshape((total_groups, m))
        out_im_2d = out_im.reshape((total_groups, m))

        groups_chunk = total_groups if total_groups <= PMAX else PMAX
        assert total_groups % groups_chunk == 0, (
            f"total_groups={total_groups} (B={B}, num_groups={num_groups}) "
            f"not divisible by chunk size {groups_chunk}"
        )
        n_partition_tiles = total_groups // groups_chunk
        C = 4097.0  # Dekker split constant for FP32

        for p in nl.affine_range(n_partition_tiles):
            p_off = p * groups_chunk
            p_end = p_off + groups_chunk

            for k in nl.affine_range(half):
                t_re = nl.load(tw_re_bcast[p_off:p_end, k : k + 1])
                t_im = nl.load(tw_im_bcast[p_off:p_end, k : k + 1])
                e_re = nl.load(x_re_2d[p_off:p_end, k : k + 1])
                e_im = nl.load(x_im_2d[p_off:p_end, k : k + 1])
                o_re = nl.load(x_re_2d[p_off:p_end, k + half : k + half + 1])
                o_im = nl.load(x_im_2d[p_off:p_end, k + half : k + half + 1])

                # NKI 0.3.0 uses nl.multiply/add/subtract explicitly; scalar-
                # tensor multiplies go through nl.multiply(scalar, tensor).
                #
                # Dekker split: x -> (xh, xl) with xh + xl == x, xh rounded.
                def _split(x):
                    xc = nl.multiply(C, x)
                    xh = nl.subtract(xc, nl.subtract(xc, x))
                    xl = nl.subtract(x, xh)
                    return xh, xl

                # twoProd(a, b) -> (hi, lo) with hi + lo == a*b (exact),
                # hi = round(a*b).
                def _two_prod(a, b):
                    ah, al = _split(a)
                    bh, bl = _split(b)
                    hi = nl.multiply(a, b)
                    ahbh_minus_hi = nl.subtract(nl.multiply(ah, bh), hi)
                    ahbl = nl.multiply(ah, bl)
                    albh = nl.multiply(al, bh)
                    albl = nl.multiply(al, bl)
                    lo = nl.add(nl.add(nl.add(ahbh_minus_hi, ahbl), albh), albl)
                    return hi, lo

                hi_rr, lo_rr = _two_prod(t_re, o_re)
                hi_ii, lo_ii = _two_prod(t_im, o_im)
                hi_ri, lo_ri = _two_prod(t_re, o_im)
                hi_ir, lo_ir = _two_prod(t_im, o_re)

                prod_re = nl.add(nl.subtract(hi_rr, hi_ii), nl.subtract(lo_rr, lo_ii))
                prod_im = nl.add(nl.add(hi_ri, hi_ir), nl.add(lo_ri, lo_ir))

                nl.store(out_re_2d[p_off:p_end, k : k + 1], value=nl.add(e_re, prod_re))
                nl.store(out_im_2d[p_off:p_end, k : k + 1], value=nl.add(e_im, prod_im))
                nl.store(
                    out_re_2d[p_off:p_end, k + half : k + half + 1],
                    value=nl.subtract(e_re, prod_re),
                )
                nl.store(
                    out_im_2d[p_off:p_end, k + half : k + half + 1],
                    value=nl.subtract(e_im, prod_im),
                )

        return out_re, out_im
else:
    # HAS_NKI False: provide a stub so the import doesn't fail on CPU-only installs.
    butterfly_stage_kernel_kahan = None  # type: ignore[assignment]
