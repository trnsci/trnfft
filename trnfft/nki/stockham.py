"""NKI kernel: radix-4 Stockham FFT stage for Trainium.

Per-stage port of the CPU reference in ``trnfft.stockham``. One kernel
invocation per stage; total log_4(N) launches per FFT vs log_2(N) for the
butterfly kernel. At N=1024 that's 5 Stockham launches vs 10 butterfly
launches — the first-order win is just the reduced launch count.

Scope note — Tensor engine vs Vector engine
--------------------------------------------
The 4x4 DFT matrix W_4 has only {1, i, -1, -i} coefficients, so the
W_4 matvec reduces to adds/subtracts plus real↔imag swaps — no actual
multiplications. The twiddle multiply IS the real compute. This POC
runs the whole stage on the Vector engine (elementwise load/mul/store
plus the W_4 linear combinations). The launch-count reduction alone
should beat butterfly at N=1024.

A follow-up (v0.13.1+) could route the twiddle step onto the Tensor
engine by absorbing per-group twiddle factors into a per-group 4x4
stationary tile — but that costs stationary reuse. Measure this POC
first before committing to that complexity.

Layout
------
At stage s with N = 4^k:
  L = 4^s                  number of already-combined 4^s-size sub-FFTs
  M = N / (4 * L)          stride within the next combined 4L-size group
  total_groups = B * L * M independent 4-point DFTs, partition-dim

Each partition row holds a single (l, m, b) 4-point transform — the 4
element values live along the free dim. Tiles are chunked at
PMAX=128 along partition dim.

Output permutation
------------------
Stockham's defining property: interleave data reorganization with
computation so no bit-reversal is needed. Stage s reads from position
``(b, l, k, m)`` in the 4D view and writes to ``(b, r, l, m)``. The
permutation happens "for free" via how we index the output HBM tensor
— no in-kernel scatter.
"""

from __future__ import annotations

from .dispatch import HAS_NKI

if HAS_NKI:
    import nki
    import nki.language as nl

    PMAX = 128

    @nki.jit
    def stockham_radix4_stage_kernel(x_re, x_im, tw_re, tw_im, n: int, stage: int):
        """Single radix-4 Stockham stage on Trainium.

        Parameters
        ----------
        x_re, x_im   : (B, N) input real/imag in HBM
        tw_re, tw_im : (B * L * M, 4) pre-broadcast twiddle factors. Host
                       computes T[l, k] = exp(-2πi * l * k / (4L)) and tiles
                       it out across all (b, l, m) partition rows so
                       elementwise multiply hits matching partition dims.
        n            : transform size (power of 4)
        stage        : stage index in [0, log_4(n))

        Returns
        -------
        out_re, out_im : (B, N) output real/imag in HBM. Indices are
                         already permuted for the next stage; no
                         host-side shuffle needed.
        """
        B, _ = x_re.shape
        L = 1 << (2 * stage)  # 4^stage
        M = n // (4 * L)
        total_groups = B * L * M

        out_re = nl.ndarray((B, n), dtype=x_re.dtype, buffer=nl.shared_hbm)
        out_im = nl.ndarray((B, n), dtype=x_im.dtype, buffer=nl.shared_hbm)

        # Input logical view: (B, L, 4, M). Flatten to (B*L*M, 4) so each
        # partition row contains one 4-point transform's data. The (b, l, m)
        # triple maps to partition index p = ((b*L)+l)*M + m, and free dim
        # holds the 4 elements of that group.
        #
        # Output logical view: (B, 4, L, M). Partition index for output-element
        # r of group (b, l, m) is ((b*4)+r)*L*M + l*M + m. The host
        # precomputes input-side reshape — output-side uses per-r strided
        # stores via explicit r-loop below.
        x_re_2d = x_re.reshape((total_groups, 4))
        x_im_2d = x_im.reshape((total_groups, 4))

        groups_chunk = total_groups if total_groups <= PMAX else PMAX
        assert total_groups % groups_chunk == 0, (
            f"total_groups={total_groups} (B={B}, L={L}, M={M}) "
            f"not divisible by chunk size {groups_chunk}"
        )
        n_partition_tiles = total_groups // groups_chunk

        # Output layout: (B, 4, L, M) flat to (B*4*L*M,). For each
        # partition row p = ((b*L)+l)*M + m of input, output-element r
        # goes to flat index b*(4*L*M) + r*(L*M) + l*M + m.
        # Equivalently: for p_in in [0, B*L*M), out_flat index for r is
        #   (p_in // (L*M)) * (4*L*M)  +  r*(L*M)  +  (p_in % (L*M))
        # Since we tile p_in in blocks of groups_chunk along partition dim,
        # we compute per-tile the corresponding output slice base per r.
        #
        # For simplicity + NKI-affine-range friendliness: handle each r
        # as its own destination-tile store. Host can also precompute
        # scatter indices into flat arrays, but this keeps everything on-chip.

        LM = L * M  # stride of a "batch row" in the output

        for p in nl.affine_range(n_partition_tiles):
            p_off = p * groups_chunk
            p_end = p_off + groups_chunk

            # Load all 4 elements of each group in this partition tile.
            x_r = nl.load(x_re_2d[p_off:p_end, :])  # (groups_chunk, 4)
            x_i = nl.load(x_im_2d[p_off:p_end, :])
            t_r = nl.load(tw_re[p_off:p_end, :])
            t_i = nl.load(tw_im[p_off:p_end, :])

            # Pre-twiddle: a[k] = x[k] * T[l, k]. Elementwise complex mul on
            # the (groups, 4) tile.
            a_r = nl.subtract(nl.multiply(x_r, t_r), nl.multiply(x_i, t_i))
            a_i = nl.add(nl.multiply(x_r, t_i), nl.multiply(x_i, t_r))

            # 4-point DFT. W_4 coefficients are {1, i, -1, -i}; the matvec
            # is all adds/subtracts plus real↔imag swaps. Select each column
            # individually — NKI can hoist this into one pass per scalar.
            a0r = a_r[:, 0:1]
            a0i = a_i[:, 0:1]
            a1r = a_r[:, 1:2]
            a1i = a_i[:, 1:2]
            a2r = a_r[:, 2:3]
            a2i = a_i[:, 2:3]
            a3r = a_r[:, 3:4]
            a3i = a_i[:, 3:4]

            # y[0] = a0 + a1 + a2 + a3
            y0r = nl.add(nl.add(a0r, a1r), nl.add(a2r, a3r))
            y0i = nl.add(nl.add(a0i, a1i), nl.add(a2i, a3i))
            # y[1] = a0 - i*a1 - a2 + i*a3
            #   -i*a1 = ( a1i, -a1r);  i*a3 = (-a3i,  a3r)
            y1r = nl.add(nl.subtract(a0r, a2r), nl.subtract(a1i, a3i))
            y1i = nl.add(nl.subtract(a0i, a2i), nl.subtract(a3r, a1r))
            # y[2] = a0 - a1 + a2 - a3
            y2r = nl.subtract(nl.add(a0r, a2r), nl.add(a1r, a3r))
            y2i = nl.subtract(nl.add(a0i, a2i), nl.add(a1i, a3i))
            # y[3] = a0 + i*a1 - a2 - i*a3
            #   i*a1 = (-a1i,  a1r); -i*a3 = ( a3i, -a3r)
            y3r = nl.add(nl.subtract(a0r, a2r), nl.subtract(a3i, a1i))
            y3i = nl.add(nl.subtract(a0i, a2i), nl.subtract(a1r, a3r))

            # Scatter into output (B, 4, L, M) layout. Partition offset for
            # output-element r of input partition row p_in is:
            #   b * (4 * L * M) + r * LM + (p_in % LM)
            # With B rows in the input having LM groups each, and tiles
            # running in slabs of groups_chunk ≤ PMAX ≤ LM for all stages
            # where LM >= PMAX. For the corner case LM < PMAX (late stages),
            # groups_chunk equals total_groups and the output index math
            # still works: partition row p_in maps to exactly one (b, l, m).
            #
            # Construct the scatter via nl.store with a pre-computed index
            # tensor — NKI's affine_range is restrictive, so build the
            # output flat view and use a 2D slice.
            #
            # Using the relation out_flat[b, r, l, m] = out[b*4*LM + r*LM + l*M + m]:
            # The partition-dim slice for this tile at output element r is a
            # stride-1 block only when the tile stays inside a single batch
            # row. Since groups_chunk ≤ PMAX ≤ LM for all B≥1, L≥1, M≥1 with
            # LM ≥ PMAX, the tile-local reshape is valid.
            out_flat_re = out_re.reshape((B * 4 * LM,))
            out_flat_im = out_im.reshape((B * 4 * LM,))
            # Compute the starting batch index for this partition tile.
            # Each batch has LM groups; tile covers rows [p_off, p_end).
            b_start = p_off // LM
            row_in_b = p_off % LM
            # (b_start, row_in_b) identifies the first (b, l, m) in tile.
            # Since groups_chunk divides LM (both powers of 2 with LM ≥ PMAX),
            # the tile lies entirely within one batch row — the output slice
            # for output-element r is contiguous.
            base = b_start * (4 * LM) + row_in_b

            nl.store(
                out_flat_re[base + 0 * LM : base + 0 * LM + groups_chunk],
                value=y0r,
            )
            nl.store(
                out_flat_im[base + 0 * LM : base + 0 * LM + groups_chunk],
                value=y0i,
            )
            nl.store(
                out_flat_re[base + 1 * LM : base + 1 * LM + groups_chunk],
                value=y1r,
            )
            nl.store(
                out_flat_im[base + 1 * LM : base + 1 * LM + groups_chunk],
                value=y1i,
            )
            nl.store(
                out_flat_re[base + 2 * LM : base + 2 * LM + groups_chunk],
                value=y2r,
            )
            nl.store(
                out_flat_im[base + 2 * LM : base + 2 * LM + groups_chunk],
                value=y2i,
            )
            nl.store(
                out_flat_re[base + 3 * LM : base + 3 * LM + groups_chunk],
                value=y3r,
            )
            nl.store(
                out_flat_im[base + 3 * LM : base + 3 * LM + groups_chunk],
                value=y3i,
            )

        return out_re, out_im
else:
    stockham_radix4_stage_kernel = None  # type: ignore[assignment]
