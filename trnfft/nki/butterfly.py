"""
NKI butterfly kernels for Trainium FFT.

Reformulation for v0.5.0: the prior scalar-loop kernel has been replaced
with a batched formulation that gathers even/odd halves into contiguous
tiles and applies the Vector Engine to the complex twiddle multiply.

For a radix-2 DIT stage `s` with group size m = 2^(s+1):
- Gather even elements: positions [k, k+m, k+2m, ...] for k in [0, half)
- Gather odd elements: positions [k+half, k+half+m, ...]
- Complex multiply: (t_re, t_im) * (o_re, o_im) element-wise
- Butterfly: (e + prod, e - prod) → write back to even/odd positions

Twiddle factors are loaded once into SBUF at the start of each stage,
then broadcast across the num_groups dimension. The inner batched ops
use Vector Engine MACs (nl.multiply + nl.add/subtract fuse into single
ALU ops in many cases).

Further optimization directions (future work):
- Promote the twiddle multiply to nisa.nc_matmul with a diagonal twiddle
  matrix, so Tensor Engine is utilized end-to-end.
- Software pipelining: overlap tile load with compute on prior tile.
- Multi-stage batching: collapse 2 or 3 radix-2 stages into one radix-4/8.
"""

from __future__ import annotations

from .dispatch import HAS_NKI

if HAS_NKI:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    @nki.jit
    def butterfly_stage_kernel(
        x_re, x_im, tw_re, tw_im, out_re, out_im,
        n: int, stage: int,
    ):
        """Batched radix-2 butterfly stage.

        Parameters
        ----------
        x_re, x_im : [n] input real/imag
        tw_re, tw_im : [half] twiddle factors for this stage
        out_re, out_im : [n] output real/imag (same shape as input)
        n : total transform size (power of 2)
        stage : stage index, 0 to log2(n)-1

        Each stage processes num_groups = n/m independent butterflies of
        size m = 2^(stage+1). All butterflies at position k within their
        groups are batched together as a Vector Engine op of length
        num_groups.
        """
        m = 1 << (stage + 1)
        half = m >> 1
        num_groups = n // m

        # Load twiddle factors for this stage (length = half)
        t_re_sbuf = nl.load(tw_re[0:half])
        t_im_sbuf = nl.load(tw_im[0:half])

        # For each butterfly position k within a group, process all groups
        # in a single batched op. This is the key change from the prior
        # scalar-loop kernel.
        for k in nl.affine_range(half):
            t_re_k = t_re_sbuf[k]
            t_im_k = t_im_sbuf[k]

            # Gather even and odd elements across all groups for position k.
            # Even indices: k, k+m, k+2m, ... (num_groups elements)
            # Odd indices:  k+half, k+half+m, ... (num_groups elements)
            # NKI stride-based load handles this gather on the partition dim.
            e_re = nl.load(x_re[k::m][:num_groups])
            e_im = nl.load(x_im[k::m][:num_groups])
            o_re = nl.load(x_re[k + half::m][:num_groups])
            o_im = nl.load(x_im[k + half::m][:num_groups])

            # Vectorized complex multiply: (t_re + i*t_im) * (o_re + i*o_im)
            # prod_re = t_re * o_re - t_im * o_im
            # prod_im = t_re * o_im + t_im * o_re
            # Vector Engine processes num_groups elements per instruction.
            prod_re = t_re_k * o_re - t_im_k * o_im
            prod_im = t_re_k * o_im + t_im_k * o_re

            # Butterfly: even = e + prod, odd = e - prod
            nl.store(out_re[k::m][:num_groups], value=e_re + prod_re)
            nl.store(out_im[k::m][:num_groups], value=e_im + prod_im)
            nl.store(out_re[k + half::m][:num_groups], value=e_re - prod_re)
            nl.store(out_im[k + half::m][:num_groups], value=e_im - prod_im)
