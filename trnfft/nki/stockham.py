"""NKI kernel: radix-4 Stockham FFT stage for Trainium.

Minimal per-stage kernel — takes pre-permuted input where each partition
row holds a single 4-point DFT group (k-axis is the free dim, stride 1).
The host driver in :func:`trnfft.fft_core._fft_via_stockham_nki` handles
the inter-stage Stockham permutation via torch permute+reshape, so the
kernel itself only has to do the twiddle multiply + 4-point DFT and
write output in the same (group, 4) layout.

Why this split
--------------
Mixing the stage-layout reshape into the kernel required a per-tile
scatter with non-contiguous indices — easy to get wrong and hard to
vectorize. Keeping the kernel as a straight elementwise-plus-add/sub
primitive makes it possible to validate in the simulator independent
of the driver logic, and defers the layout question to the part of the
code that can use torch's native permute/contiguous for free.

Cost note: the driver's per-stage permute+contiguous triggers an HBM
shuffle on hardware. For this POC that's acceptable — we're measuring
whether log_4(N) launches beat log_2(N), and permutes are a fixed cost.
A fused-stage kernel (Thread C) would eliminate those shuffles.

Input / output layout
---------------------
- ``x_re, x_im`` : ``(total_groups, 4)`` — each partition row is one
  4-point DFT group, k-axis is the free dim.
- ``tw_re, tw_im`` : ``(total_groups, 4)`` — pre-broadcast twiddle
  factors. Host computes ``T[l, k] = exp(-2πi * l * k / (4L))`` and
  tiles across all (b, l, m) partition rows.
- ``out_re, out_im`` : ``(total_groups, 4)`` — 4-point DFT result for
  each group, same layout.

W_4 matvec
----------
W_4 has coefficients {1, i, -1, -i}, so the matvec reduces to adds and
real/imag swaps — no actual multiplications. Runs on the Vector engine.
The twiddle multiply IS the real compute on the Vector engine. A
follow-up could absorb twiddles into per-group 4x4 stationary tiles on
the Tensor engine, but that costs stationary reuse — measure the
launch-count win first.
"""

from __future__ import annotations

from .dispatch import HAS_NKI

if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    PMAX = 128

    @nki.jit
    def stockham_radix4_stage_kernel(x_re, x_im, tw_re, tw_im):
        """Single radix-4 Stockham stage: pre-twiddle + 4-point DFT.

        All four arguments are ``(total_groups, 4)`` tensors in HBM.
        Returns ``(total_groups, 4)`` post-stage tensors; driver handles
        the inter-stage permutation.
        """
        total_groups, _ = x_re.shape

        out_re = nl.ndarray((total_groups, 4), dtype=x_re.dtype, buffer=nl.shared_hbm)
        out_im = nl.ndarray((total_groups, 4), dtype=x_im.dtype, buffer=nl.shared_hbm)

        groups_chunk = total_groups if total_groups <= PMAX else PMAX
        assert total_groups % groups_chunk == 0, (
            f"total_groups={total_groups} not divisible by chunk size {groups_chunk}"
        )
        n_partition_tiles = total_groups // groups_chunk

        for p in nl.affine_range(n_partition_tiles):
            p_off = p * groups_chunk
            p_end = p_off + groups_chunk

            x_r = nl.load(x_re[p_off:p_end, :])  # (groups_chunk, 4)
            x_i = nl.load(x_im[p_off:p_end, :])
            t_r = nl.load(tw_re[p_off:p_end, :])
            t_i = nl.load(tw_im[p_off:p_end, :])

            # Pre-twiddle: a[k] = x[k] * T[l, k] (complex mul, elementwise).
            a_r = nl.subtract(nl.multiply(x_r, t_r), nl.multiply(x_i, t_i))
            a_i = nl.add(nl.multiply(x_r, t_i), nl.multiply(x_i, t_r))

            a0r = a_r[:, 0:1]
            a0i = a_i[:, 0:1]
            a1r = a_r[:, 1:2]
            a1i = a_i[:, 1:2]
            a2r = a_r[:, 2:3]
            a2i = a_i[:, 2:3]
            a3r = a_r[:, 3:4]
            a3i = a_i[:, 3:4]

            # W_4 matvec (no multiplications — {1, i, -1, -i} coefficients).
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

            nl.store(out_re[p_off:p_end, 0:1], value=y0r)
            nl.store(out_im[p_off:p_end, 0:1], value=y0i)
            nl.store(out_re[p_off:p_end, 1:2], value=y1r)
            nl.store(out_im[p_off:p_end, 1:2], value=y1i)
            nl.store(out_re[p_off:p_end, 2:3], value=y2r)
            nl.store(out_im[p_off:p_end, 2:3], value=y2i)
            nl.store(out_re[p_off:p_end, 3:4], value=y3r)
            nl.store(out_im[p_off:p_end, 3:4], value=y3i)

        return out_re, out_im

    # ---------------------------------------------------------------------------
    # Thread C phase 2 — fused multi-stage kernel (NOT YET IMPLEMENTED).
    #
    # Intent: run all log_4(N) Stockham stages in a single NKI kernel call,
    # keeping intermediate results in SBUF between stages to eliminate the
    # per-stage HBM round-trip that the driver's gather-based pack/unpack still
    # incurs (Thread C phase 1 reduced XLA graph ops but not HBM traffic).
    #
    # Architectural constraint: for dispatched N values (N > 256, so
    # total_groups = B*N/4 > 64), the inter-stage permutation scatters elements
    # across partition tiles (each tile covers PMAX=128 consecutive groups).
    # NKI's affine_range model does not support cross-tile indirect addressing;
    # a cross-tile gather would require nki.language.gather with runtime indices,
    # which as of NKI 0.3.0 is not validated for this access pattern.
    #
    # Unblocking conditions:
    #   1. NKI gains a cross-tile indirect-load primitive, OR
    #   2. The kernel is restructured to work in a block layout where inter-stage
    #      permutations are intra-tile (requires reordering twiddles, not data).
    # ---------------------------------------------------------------------------
    def stockham_radix4_fused_kernel(*args, **kwargs):
        raise NotImplementedError(
            "stockham_radix4_fused_kernel (Thread C phase 2) is not yet implemented. "
            "Use stockham_radix4_stage_kernel via _fft_via_stockham_nki instead."
        )

    # ---------------------------------------------------------------------------
    # Thread B: radix-8 stage kernel — Tensor-engine W_8 matmul
    #
    # Replaces the radix-4 Vector-engine add/sub W_4 butterfly with an
    # nc_matmul on the 8×8 DFT matrix W_8.  W_8 entries are non-trivial
    # (±√2/2 ± i√2/2), so the Tensor engine earns its keep here unlike W_4.
    #
    # Kernel structure (two phases in one kernel call):
    #
    #   Phase 1 — twiddle multiply (Vector engine):
    #     Same as radix-4: element-wise complex multiply per (groups_chunk, 8)
    #     tile.  Result written to an HBM scratch buffer.
    #
    #   Phase 2 — W_8 matmul (Tensor engine):
    #     Reads scratch via nl.load_transpose2d (HBM → transposed SBUF tile)
    #     then accumulates 4 nc_matmul calls (standard complex GEMM pattern).
    #
    #   The HBM scratch round-trip between phases is the cost of using
    #   nl.load_transpose2d (NKI requires HBM source for transpose loads).
    #   Hardware bench will determine if the Tensor-engine W_8 win plus
    #   fewer stages (log_8(N) vs log_4(N)) outweighs the scratch cost.
    #
    # Input/output layout
    # --------------------
    # - x_re, x_im, tw_re, tw_im : (total_groups, 8) HBM
    # - w8_re, w8_im              : (8, 8) HBM — shared W_8 DFT matrix
    #   W_8[k, n] = exp(-2πi·k·n/8). Symmetric so W_8^T = W_8; stored as
    #   (K=8, N=8) to match the nc_matmul moving-tile convention.
    # - out_re, out_im            : (total_groups, 8) HBM
    # ---------------------------------------------------------------------------

    @nki.jit
    def stockham_radix8_stage_kernel(x_re, x_im, tw_re, tw_im, w8_re, w8_im):
        """Radix-8 Stockham stage: twiddle (Vector) + W_8 (Tensor engine).

        All x/tw tensors are ``(total_groups, 8)``; w8 tensors are ``(8, 8)``.
        Returns ``(total_groups, 8)`` post-stage tensors.
        """
        total_groups, _ = x_re.shape

        out_re = nl.ndarray((total_groups, 8), dtype=x_re.dtype, buffer=nl.shared_hbm)
        out_im = nl.ndarray((total_groups, 8), dtype=x_im.dtype, buffer=nl.shared_hbm)

        # Scratch buffers: twiddle-applied data stored here so Phase 2 can
        # read it back via nl.load_transpose2d (requires HBM source).
        scratch_re = nl.ndarray((total_groups, 8), dtype=x_re.dtype, buffer=nl.shared_hbm)
        scratch_im = nl.ndarray((total_groups, 8), dtype=x_im.dtype, buffer=nl.shared_hbm)

        groups_chunk = total_groups if total_groups <= PMAX else PMAX
        assert total_groups % groups_chunk == 0, (
            f"total_groups={total_groups} not divisible by chunk size {groups_chunk}"
        )
        n_partition_tiles = total_groups // groups_chunk

        # ── Phase 1: twiddle multiply (Vector engine) ──────────────────────
        for p in nl.affine_range(n_partition_tiles):
            p_off = p * groups_chunk
            p_end = p_off + groups_chunk

            x_r = nl.load(x_re[p_off:p_end, :])  # (groups_chunk, 8)
            x_i = nl.load(x_im[p_off:p_end, :])
            t_r = nl.load(tw_re[p_off:p_end, :])
            t_i = nl.load(tw_im[p_off:p_end, :])

            a_r = nl.subtract(nl.multiply(x_r, t_r), nl.multiply(x_i, t_i))
            a_i = nl.add(nl.multiply(x_r, t_i), nl.multiply(x_i, t_r))

            nl.store(scratch_re[p_off:p_end, :], a_r)
            nl.store(scratch_im[p_off:p_end, :], a_i)

        # ── Phase 2: W_8 matmul (Tensor engine) ────────────────────────────
        # nc_matmul shape convention: stationary (K, M), moving (K, N) → PSUM (M, N)
        # Here K=8 (contraction), M=groups_chunk (partition/batch), N=8 (output).
        TILE_K = 8
        TILE_N = 8

        # W_8 is fixed across all partition tiles; load once outside the loop.
        w8_r = nl.load(w8_re[:TILE_K, :TILE_N])  # (8, 8)
        w8_i = nl.load(w8_im[:TILE_K, :TILE_N])
        neg_w8_i = nl.negative(w8_i)

        for m in nl.affine_range(n_partition_tiles):
            m_off = m * groups_chunk

            psum_cr = nl.zeros((groups_chunk, TILE_N), dtype=nl.float32, buffer=nl.psum)
            psum_ci = nl.zeros((groups_chunk, TILE_N), dtype=nl.float32, buffer=nl.psum)

            # Load scratch with transpose: (groups_chunk, 8) → (K=8, groups_chunk)
            ar_t = nl.load_transpose2d(scratch_re[m_off : m_off + groups_chunk, :TILE_K])
            ai_t = nl.load_transpose2d(scratch_im[m_off : m_off + groups_chunk, :TILE_K])

            # C_real = A_real @ W_8_real − A_imag @ W_8_imag
            nisa.nc_matmul(dst=psum_cr, stationary=ar_t, moving=w8_r, accumulate=True)
            nisa.nc_matmul(dst=psum_cr, stationary=ai_t, moving=neg_w8_i, accumulate=True)

            # C_imag = A_real @ W_8_imag + A_imag @ W_8_real
            nisa.nc_matmul(dst=psum_ci, stationary=ar_t, moving=w8_i, accumulate=True)
            nisa.nc_matmul(dst=psum_ci, stationary=ai_t, moving=w8_r, accumulate=True)

            cr_sbuf = nl.ndarray((groups_chunk, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            ci_sbuf = nl.ndarray((groups_chunk, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cr_sbuf, src=psum_cr)
            nisa.tensor_copy(dst=ci_sbuf, src=psum_ci)
            if x_re.dtype != nl.float32:
                cr_sbuf = nl.cast(cr_sbuf, dtype=x_re.dtype)
                ci_sbuf = nl.cast(ci_sbuf, dtype=x_re.dtype)
            nl.store(out_re[m_off : m_off + groups_chunk, :], value=cr_sbuf)
            nl.store(out_im[m_off : m_off + groups_chunk, :], value=ci_sbuf)

        return out_re, out_im

else:
    stockham_radix4_stage_kernel = None  # type: ignore[assignment]
    stockham_radix4_fused_kernel = None  # type: ignore[assignment]
    stockham_radix8_stage_kernel = None  # type: ignore[assignment]
