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
    def butterfly_stage_kernel(x_re, x_im, tw_re_bcast, tw_im_bcast, n: int, stage: int):
        """Batched radix-2 butterfly stage with 2D partition layout.

        Parameters
        ----------
        x_re, x_im                 : [n] input real/imag (HBM)
        tw_re_bcast, tw_im_bcast   : [num_groups, half] twiddle factors broadcast
                                     across the num_groups (partition) dim by the host.
                                     Row g column k is `cos/sin(±2π·k / m)`. Pre-broadcasting
                                     is needed because NKI 2.24 element-wise ops require
                                     matching partition dims and don't auto-broadcast (1,1)
                                     tiles to (num_groups, 1).
        n                          : transform size (power of 2)
        stage                      : stage index in [0, log2(n))

        Returns
        -------
        out_re, out_im : [n] output real/imag (HBM, allocated by the kernel)
        """
        m = 1 << (stage + 1)
        half = m >> 1
        num_groups = n // m

        out_re = nl.ndarray((n,), dtype=x_re.dtype, buffer=nl.shared_hbm)
        out_im = nl.ndarray((n,), dtype=x_im.dtype, buffer=nl.shared_hbm)

        # 2D views. Partition dim = num_groups (the gather dim).
        x_re_2d = x_re.reshape((num_groups, m))
        x_im_2d = x_im.reshape((num_groups, m))
        out_re_2d = out_re.reshape((num_groups, m))
        out_im_2d = out_im.reshape((num_groups, m))

        # NKI affine_range can't evaluate min() symbolically, so use a constant
        # chunk size. For all power-of-2 transforms with num_groups <= 128 this
        # is one tile; otherwise num_groups is always a multiple of 128, so the
        # division is exact and no tail tile is needed.
        groups_chunk = num_groups if num_groups <= PMAX else PMAX
        assert num_groups % groups_chunk == 0, \
            f"num_groups={num_groups} not divisible by chunk size {groups_chunk}"
        n_partition_tiles = num_groups // groups_chunk

        for p in nl.affine_range(n_partition_tiles):
            p_off = p * groups_chunk
            p_end = p_off + groups_chunk  # constant slice size

            # Process each butterfly position k within this partition tile.
            for k in nl.affine_range(half):
                # Twiddle for this column, broadcast across the partition (groups) dim.
                t_re_col = nl.load(tw_re_bcast[p_off:p_end, k:k+1])
                t_im_col = nl.load(tw_im_bcast[p_off:p_end, k:k+1])

                # Even and odd columns.
                e_re = nl.load(x_re_2d[p_off:p_end, k:k+1])
                e_im = nl.load(x_im_2d[p_off:p_end, k:k+1])
                o_re = nl.load(x_re_2d[p_off:p_end, k+half:k+half+1])
                o_im = nl.load(x_im_2d[p_off:p_end, k+half:k+half+1])

                # Complex multiply: (t_re + i*t_im) * (o_re + i*o_im)
                prod_re = t_re_col * o_re - t_im_col * o_im
                prod_im = t_re_col * o_im + t_im_col * o_re

                # Butterfly: even = e + prod, odd = e - prod
                nl.store(out_re_2d[p_off:p_end, k:k+1], value=e_re + prod_re)
                nl.store(out_im_2d[p_off:p_end, k:k+1], value=e_im + prod_im)
                nl.store(out_re_2d[p_off:p_end, k+half:k+half+1], value=e_re - prod_re)
                nl.store(out_im_2d[p_off:p_end, k+half:k+half+1], value=e_im - prod_im)

        return out_re, out_im
