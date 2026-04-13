# FFT on Trainium is a GEMM problem

## The observation

At small N on Trainium, one N×N DFT matmul on the Tensor engine beats
log₂(N) radix-2 butterfly stages on the Vector engine by 2–4.5×.

Numbers from trn1 (NKI 2.24.5133.0), mean over ~400 runs each, forward FFT
of random FP32 signal. Butterfly column forces `_DFT_GEMM_THRESHOLD = 0`;
DFT-GEMM column forces threshold to ∞. Both paths hit the same
user-facing `trnfft.fft`.

| N   | DFT-GEMM (μs) | Butterfly (μs) | Speedup |
| --- | ------------- | -------------- | ------- |
| 8   | 1717          | 3779           | 2.2×    |
| 16  | 1702          | 4860           | 2.9×    |
| 32  | 1680          | 5930           | 3.5×    |
| 64  | 1839          | 7086           | 3.9×    |
| 128 | 1856          | 8399           | 4.5×    |

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

The butterfly-to-DFT-GEMM speedup is still *growing* at N=128 (2.2× → 4.5×
across the sweep). The actual crossover is above 128. We haven't measured
it because the current DFT-GEMM kernel parameters (one PSUM tile,
TILE_K=128) max out at N=128. Extending the kernel to multi-tile K for
N ∈ {256, 512, 1024} is a straightforward generalization — the crossover
measurement falls out of that.

## What this means for the threshold

Today `_DFT_GEMM_THRESHOLD = 128`, a conservative "fits one tile" choice.
The data says that's **not the right cutoff** — it's the matmul-kernel
geometry limit, not the architectural crossover. Proper threshold
derivation is v0.12 milestone 2: extend DFT-GEMM past 128 and read the
crossover off a widened sweep.

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

Data collected v0.11.0+6fe841c, trn1, 2026-04-13.
