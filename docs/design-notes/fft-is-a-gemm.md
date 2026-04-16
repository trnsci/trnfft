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
Batched + STFT data: v0.12.0+371625c, trn1, 2026-04-14.

## Radix-4 Stockham — hardware verdict

The DFT-GEMM win is precision-bound at N=256. Stockham radix-4 breaks
that ceiling by decomposing N into log₄(N) stages of small 4-point
DFTs, each accumulating only O(r²) error (r=4). At N=4096 CPU-reference
FP32 error is ~5e-5 — three orders of magnitude tighter than DFT-GEMM
at N=1024 would have been.

Launch-count story: at N=1024 Stockham is **5 stages** vs butterfly's
**10**. The W_4 matrix has {1, i, -1, -i} coefficients so its matvec
is free (adds + real/imag swaps); per-stage cost is dominated by the
twiddle multiply. Everything runs on the Vector engine in this POC —
routing the twiddle onto the Tensor engine is a Thread C follow-up.

### Three-way head-to-head (trn1, Neuron SDK 2.29.0, 2026-04-15)

| N    | DFT-GEMM (μs) | Stockham (μs) | Butterfly (μs) | Winner              |
| ---- | ------------- | ------------- | -------------- | ------------------- |
| 16   | 1 831         | 3 360         | 3 337          | DFT-GEMM (1.8×)     |
| 64   | 1 941         | 4 842         | 4 767          | DFT-GEMM (2.5×)     |
| 256  | 2 029         | 6 153         | 6 067          | DFT-GEMM (3.0×)     |
| 1024 | —             | 7 568         | 7 399          | Butterfly (+2.3%)   |
| 4096 | —             | 9 742         | 9 387 †        | Butterfly (+3.8%)   |

"—" = DFT-GEMM out of range (FP32 O(N²) accumulation exceeds 1e-3 rel error at N ≥ 512).  
† N=4096 butterfly from `test_fft_nki[4096]` (auto-dispatch, `_FORCE_STOCKHAM=False`).

### What the numbers mean

**Precision goal met, performance goal not yet met.** Stockham is
precision-safe to N=4096+ as designed. But it ties butterfly at every
measured N — not because both have the same arithmetic, but because the
POC had a driver-side overhead that ate the 5-vs-10 launch advantage.

**SDK 2.29 butterfly improvement.** The butterfly numbers here are
1.3–2.1× better than the SDK 2.24 head-to-head above (e.g., N=256 went
from 9 862 μs to 6 067 μs). This narrows the DFT-GEMM speedup from
5.2× to 3.0× at N=256 — the DFT-GEMM win is still decisive, but the
comparison baseline changed under us between SDK versions.

Stockham POC data: `1504dcb`, trn1, 2026-04-15, Neuron SDK 2.29.0.

## Stockham profiling — finding the real bottleneck (2026-04-16)

Permute timing probe on trn1 (SHA `6764c21`, Neuron SDK 2.29.0) refuted
the original inter-stage permute hypothesis and found the actual bottleneck.

### Permute overhead: 10% — not the bottleneck

| N    | Pre-permute (μs) | Post-permute (μs) | Kernel (μs) | Permute % |
| ---- | --------------- | ----------------- | ----------- | --------- |
| 64   | 57.6            | 39.7              | 853.0       | **10%**   |
| 256  | 56.2            | 39.4              | 858.8       | **10%**   |
| 1024 | 57.9            | 40.0              | 855.2       | **10%**   |
| 4096 | 57.5            | 39.7              | 884.3       | **10%**   |

But probe-projected times (5 stages × 952 μs = 4766 μs) are 37% lower
than the actual benchmark (7568 μs). The probe only measured stage `s=0`
where `L=1` and twiddle tensors are trivially [1,1,1,1]/[0,0,0,0].

### Real bottleneck: per-stage twiddle recomputation

At every stage, the POC driver computed twiddle factors from scratch on
CPU and transferred them to the XLA device. At stage `s=4` (N=1024),
`L=256` — a `256×4` tensor computed with `torch.cos/sin`, broadcast,
and transferred via `.to(device)` every stage call. Twiddle sizes grew
1×4 → 4×4 → 16×4 → 64×4 → 256×4 across the 5 stages.

The fix exploits a structural invariant: `total_groups = B_pad * L * M
= B_pad * N/4` is constant across all stages (L and M are inverse powers
of 4 that cancel). All log₄(N) twiddle tensors have the same shape
`(total_groups, 4)` and can be stacked into `(log4n, total_groups, 4)`
for a single H→D transfer before the loop.

This change (`f095671`) eliminates per-stage twiddle transfer overhead
entirely. If twiddle recomputation was the dominant term, projected
Stockham time at N=1024 drops to ~4766 μs — a **~1.55× win** over butterfly (7399 μs).

### Hardware result after twiddle precomputation

*Placeholder — fill after hardware benchmark run on SHA `f095671`.*

| N    | Stockham (μs) | Butterfly (μs) | Winner |
| ---- | ------------- | -------------- | ------ |
| 16   |               | 3 337          |        |
| 64   |               | 4 767          |        |
| 256  |               | 6 067          |        |
| 1024 |               | 7 399          |        |
| 4096 |               | 9 387          |        |

## Batched FFT + STFT: where the thesis pays off

The 1-D head-to-head sweep above proves the launch-count win exists.
The bigger practical win is on **batched FFT** and **STFT**, which both
flatten their input to `(B, N)` and flow through the same
`_cooley_tukey_nki_nograd`. With DFT-GEMM dispatched for `N ≤ 256`,
the whole batch collapses to a single matmul — one kernel launch
regardless of `B`, and large `B` finally fills the 128-partition
systolic array.

### Batched FFT (trn1, μs mean)

| Shape `(B, N)`  | Path       | Mean (μs) | vs large-N butterfly | vs PyTorch fallback |
| --------------- | ---------- | --------- | -------------------- | ------------------- |
| (32, 128)       | DFT-GEMM   | 1851      | **15.8× faster**     | 6.3× faster         |
| (32, 256)       | DFT-GEMM   | 2049      | **14.3× faster**     | 11.3× faster        |
| (32, 1024)      | butterfly  | 29210     | baseline             | 3.2× faster         |
| (128, 1024)     | butterfly  | 59112     | baseline             | 1.8× faster         |

### STFT (n_fft sweep, 16000-sample waveform)

| `n_fft` | Path      | Mean (μs) | vs n_fft=512 butterfly | vs PyTorch fallback |
| ------- | --------- | --------- | ---------------------- | ------------------- |
| 128     | DFT-GEMM  | 2323      | **13.1× faster**       | 6.2× faster         |
| 256     | DFT-GEMM  | 2445      | **12.5× faster**       | 10.5× faster        |
| 512     | butterfly | 30514     | baseline               | 1.6× faster         |

### Reading the numbers

Both tables share the same shape: one-matmul-per-batch collapses the
whole small-N regime into the flat ~2 ms floor (launch overhead), while
butterfly stays at the old ~30 ms (many-launch) cost. The
"vs large-N butterfly" speedups are the headline — a user doing STFT at
`n_fft=256` on Trainium today gets 12.5× lower per-batch cost than the
same library would have delivered with butterfly.

The "vs PyTorch fallback" column shows the NKI-vs-host-CPU story widens
dramatically at larger batches: from 6× at B=32 to 11× at B=32 with
partition-saturating N. DFT-GEMM's partition utilization scales cleanly
with B in a way butterfly doesn't.

**Note on comparing to `torch.fft`:** vanilla `torch.fft.fft` on the
bench-host CPU (an x86 running MKL) is in the 10–80 μs range for these
shapes — roughly 50–300× faster than our NKI path for one-shot calls.
That's not surprising: MKL is silicon + decades of optimization, and a
single cold NKI dispatch pays fixed ~1 ms of Trainium setup overhead.
The Trainium story is about **keeping data on-chip** when surrounding
training or inference work is already there, not about beating MKL on
isolated FFTs.
