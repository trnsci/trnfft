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
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

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
