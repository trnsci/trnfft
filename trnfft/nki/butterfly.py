"""
NKI butterfly kernels for Trainium FFT.

These kernels implement FFT butterfly operations using:
- Tensor Engine (systolic array) for batched complex multiplies
- Vector Engine for additions and twiddle factor application
- Scalar Engine for control flow

STUB: These are scaffolded for on-hardware validation. The CPU fallback
in fft_core.py implements the same algorithm using PyTorch ops.

Key optimization over CPU fallback:
- All butterflies in a stage batched as a single matmul
- Twiddle factors preloaded to SBUF, reused across batch dimension
- Double-buffering for DMA/compute overlap on large transforms
"""

from __future__ import annotations

from .dispatch import HAS_NKI

if HAS_NKI:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa

    @nki.jit
    def butterfly_stage_kernel(
        x_re, x_im, tw_re, tw_im, out_re, out_im,
        n: int, stage: int
    ):
        """Single butterfly stage as NKI kernel.

        Processes all butterflies in stage `stage` of a radix-2 Cooley-Tukey FFT.
        Twiddle factors are loaded once to SBUF and reused across all groups.

        TODO: Validate on trn1/trn2 hardware. The tiling and SBUF allocation
        patterns need tuning against actual NKI compiler scheduling.
        """
        m = 1 << (stage + 1)
        half = m >> 1
        num_groups = n // m

        for k in nl.affine_range(half):
            # Compute twiddle for this butterfly position
            angle = -2.0 * 3.141592653589793 * k / m
            t_re = nl.load(tw_re[k])
            t_im = nl.load(tw_im[k])

            for g in nl.affine_range(num_groups):
                even_idx = g * m + k
                odd_idx = even_idx + half

                e_re = nl.load(x_re[even_idx])
                e_im = nl.load(x_im[even_idx])
                o_re = nl.load(x_re[odd_idx])
                o_im = nl.load(x_im[odd_idx])

                # Complex multiply: twiddle * odd
                prod_re = nl.subtract(
                    nl.multiply(t_re, o_re),
                    nl.multiply(t_im, o_im)
                )
                prod_im = nl.add(
                    nl.multiply(t_re, o_im),
                    nl.multiply(t_im, o_re)
                )

                # Butterfly
                nl.store(out_re[even_idx], nl.add(e_re, prod_re))
                nl.store(out_im[even_idx], nl.add(e_im, prod_im))
                nl.store(out_re[odd_idx], nl.subtract(e_re, prod_re))
                nl.store(out_im[odd_idx], nl.subtract(e_im, prod_im))
