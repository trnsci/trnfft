"""
NKI butterfly kernel for radix-2 Cooley-Tukey FFT on Trainium.

NKI 2.24+ requires that any SBUF tile be 2D with the first dimension being
the partition dim, and partition size ≤ 128. This kernel reshapes the 1D
input as a 2D `(num_groups, m)` array — each row is one butterfly group
of size m. The partition dim becomes num_groups, so we can vectorize across
all groups in a single Vector Engine instruction per butterfly position.

When num_groups > 128 (early stages of large transforms), we tile along the
partition dim in chunks of ≤ 128.

Validated against neuronxcc 2.24.5133.0 on Deep Learning AMI Neuron PyTorch
2.9 (Ubuntu 24.04).
"""

from __future__ import annotations

from .dispatch import HAS_NKI

if HAS_NKI:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    PMAX = 128

    @nki.jit
    def butterfly_stage_kernel(
        x_re, x_im, tw_re, tw_im, out_re, out_im,
        n: int, stage: int,
    ):
        """Batched radix-2 butterfly stage with 2D partition layout.

        Parameters
        ----------
        x_re, x_im     : [n] input real/imag (HBM)
        tw_re, tw_im   : [half] twiddle factors for this stage (HBM)
        out_re, out_im : [n] output real/imag (HBM, same shape as input)
        n              : transform size (power of 2)
        stage          : stage index in [0, log2(n))

        Layout
        ------
        Reshape the 1D input as 2D (num_groups, m). Row g column j of this
        view is element g*m + j of the 1D input — the j-th element of group g.

        For each butterfly position k in [0, half):
            even column = column k     (one element per group)
            odd column  = column k+half
        Twiddle for position k is a scalar across all groups.

        We load the even/odd columns as (num_groups, 1) SBUF tiles. The first
        dim is the partition dim, satisfying the NKI 2.24 constraint.

        For num_groups > 128, tile the partition dim in chunks of 128.
        """
        m = 1 << (stage + 1)
        half = m >> 1
        num_groups = n // m

        # 2D views. Partition dim = num_groups (the gather dim).
        x_re_2d = x_re.reshape((num_groups, m))
        x_im_2d = x_im.reshape((num_groups, m))
        out_re_2d = out_re.reshape((num_groups, m))
        out_im_2d = out_im.reshape((num_groups, m))

        # Tile over num_groups in chunks of PMAX.
        n_partition_tiles = (num_groups + PMAX - 1) // PMAX

        for p in nl.affine_range(n_partition_tiles):
            p_off = p * PMAX
            p_end = min(p_off + PMAX, num_groups)

            # Process each butterfly position k within this partition tile.
            for k in nl.affine_range(half):
                # Twiddle is a single complex scalar for all groups at position k.
                t_re_k = nl.load(tw_re[k:k+1])  # shape (1,) → broadcastable
                t_im_k = nl.load(tw_im[k:k+1])

                # Even and odd columns. Each load yields a (groups_chunk, 1) tile
                # — first dim is partition dim, satisfying NKI constraints.
                e_re = nl.load(x_re_2d[p_off:p_end, k:k+1])
                e_im = nl.load(x_im_2d[p_off:p_end, k:k+1])
                o_re = nl.load(x_re_2d[p_off:p_end, k+half:k+half+1])
                o_im = nl.load(x_im_2d[p_off:p_end, k+half:k+half+1])

                # Complex multiply: (t_re + i*t_im) * (o_re + i*o_im)
                prod_re = t_re_k * o_re - t_im_k * o_im
                prod_im = t_re_k * o_im + t_im_k * o_re

                # Butterfly: even = e + prod, odd = e - prod
                nl.store(out_re_2d[p_off:p_end, k:k+1], value=e_re + prod_re)
                nl.store(out_im_2d[p_off:p_end, k:k+1], value=e_im + prod_im)
                nl.store(out_re_2d[p_off:p_end, k+half:k+half+1], value=e_re - prod_re)
                nl.store(out_im_2d[p_off:p_end, k+half:k+half+1], value=e_im - prod_im)
