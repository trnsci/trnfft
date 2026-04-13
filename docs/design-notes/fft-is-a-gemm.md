# FFT on Trainium is a GEMM problem

## The observation

At small N on Trainium, one N×N DFT matmul on the Tensor engine beats
log₂(N) radix-2 butterfly stages on the Vector engine by 2–4.5×.

Numbers from trn1 (NKI 2.24.5133.0), mean over ~400 runs each, forward FFT
of random FP32 signal. Butterfly column forces `_DFT_GEMM_THRESHOLD = 0`;
DFT-GEMM column forces threshold to ∞. Both paths hit the same
user-facing `trnfft.fft`.

| N    | DFT-GEMM (μs) | Butterfly (μs) | Speedup |
| ---- | ------------- | -------------- | ------- |
| 8    | 1716          | 3832           | 2.2×    |
| 16   | 1687          | 4762           | 2.8×    |
| 32   | 1679          | 5826           | 3.5×    |
| 64   | 1833          | 6997           | 3.8×    |
| 128  | 1806          | 8316           | 4.6×    |
| 256  | 1882          | 9862           | 5.2×    |
| 512  | 2126          | 12142          | 5.7×    |
| 1024 | 2954          | 15746          | 5.3×    |
| 2048 | 17500         | 23819          | 1.4×    |

## Why the butterfly path loses

The Cooley-Tukey NKI path issues **one kernel launch per stage**. A radix-2
DIT FFT at N has log₂(N) stages: 3 for N=8, 7 for N=128. Each launch pays
the XLA dispatch cost and does a partition-tiled pass over `x`. Wall-clock
grows linearly in log₂(N) even though the arithmetic per stage does not
dominate — which you can see directly in the butterfly column: a clean
~1 ms per stage.

Meanwhile the DFT-GEMM path is **one launch**, period. `complex_gemm`
routes `x @ W` onto the Tensor engine via `nisa.nc_matmul` + PSUM
accumulation. The actual matmul work is tiny at N ≤ 128 (the N×N DFT
matrix fits inside a single PSUM tile), so total time is dominated by the
fixed launch overhead — hence the flat ~1.7 ms column.

## Why this is Trainium-specific

On a CPU the same arithmetic argument would go the other way: launches are
cheap and `O(N²)` vs `O(N log N)` matters. On a GPU the crossover sits at
smaller N because launch overhead is lower. On Trainium specifically, (a)
NKI kernel launches are expensive relative to small-N matmul work and (b)
the Tensor engine is uniquely cheap for matrix work — so both effects
point at the GEMM formulation at small N.

This is why Trainium won't just be served by porting FFT libraries. The
launch profile is different enough that algorithm choice has to be
re-derived from hardware.

## Where the crossover really is

Widened sweep to N=2048 (see widened table above) lands three findings
the original N≤128 probe couldn't see:

1. **DFT-GEMM is roughly constant at ~1.7-2.1 ms through N=512**, then
   starts climbing. Consistent with "one matmul, launch overhead
   dominates at small N". The `complex_gemm` kernel's K-tiling doesn't
   add much cost until K gets large enough that the K-loop body actually
   amortizes the dispatch setup.

2. **Butterfly grows cleanly linear in log₂(N)**: each new stage adds
   about 1–1.2 ms regardless of N. That's the per-launch cost, visible
   in isolation.

3. **Hard cliff at N=2048**: DFT-GEMM jumps from 2954 μs at N=1024 to
   17500 μs — a 6× jump for a 2× size increase. TILE_K=128 means
   16 K-tile iterations at N=2048 vs 8 at N=1024; combined with M=1
   partition underutilization (single-FFT case wastes 127 of 128
   partitions), this is where the O(N²) arithmetic actually starts
   to bite.

The **performance** crossover is N ≈ 1024–2048. The **precision**
crossover is N=256 — above that, FP32 `nc_matmul` accumulation exceeds
1e-3 relative error (measured ~2.2% at N=1024 under the widened probe).

## What this means for the threshold

Precision binds before performance does. `_DFT_GEMM_THRESHOLD = 256`
pins the path to the regime where FP32 is safe. Raising it further
requires breaking the O(N²) accumulation — Thread B (Stockham radix-r)
keeps compute on the Tensor engine via `r×r` stage matmuls without
paying quadratic accumulation, and is the structurally-correct path to
extend the DFT-GEMM win past N=256 into the N=512–1024 region where
butterfly is still 5× slower.

## What this means for the library

1. **Batched FFT** and **STFT** are obvious follow-ups. They're both
   "many FFTs at small N" and should collapse to a single matmul
   (`W @ X` where X is the stacked batch/frames matrix). No per-FFT
   launch overhead at all.

2. **Medium-N** (256–2048) via Stockham radix-r, where each stage is a
   small GEMM of size (r, r) ⊗ identity applied across the input — still
   on the Tensor engine, still one launch per stage, but now the stage
   count drops from log₂(N) to log_r(N). With r=8 and N=1024, that's
   one stage's work distributed across the partition dim — probably a
   single `nc_matmul` call per stage on carefully-tiled input.

3. **Large-N** may still want butterfly, but with SBUF-resident twiddles
   and a persistent kernel that fuses multiple stages into one launch.
   Closes the log₂(N) launch tax from the other direction.

The unifying thesis: **FFT on Trainium is a problem about kernel launches
and Tensor-engine utilization, not arithmetic complexity**. Every design
choice should be judged against those two axes.

## Version

Small-N data (N ≤ 128): v0.11.0+6fe841c, trn1, 2026-04-13.
Widened data (N ≤ 2048): v0.11.0+03d3ac2, trn1, 2026-04-13.
