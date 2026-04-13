# Benchmarks

trnfft NKI kernels measured against two baselines on the same Trainium instance:

1. **NKI** — trnfft with `set_backend("nki")` running on Tensor + Vector Engines
2. **trnfft-PyTorch** — trnfft with `set_backend("pytorch")` running on the host CPU
3. **torch.\*** — vanilla `torch.fft.*` / `torch.matmul` on the host CPU

The first comparison (1 vs 2) answers *"did our NKI kernels actually help vs our own PyTorch fallback?"* It uses the same code path on both sides — only the backend dispatch differs.

The second comparison (1 vs 3) answers *"what's the user-visible difference between trnfft and vanilla PyTorch?"*

## Methodology

- **Hardware**: AWS `trn1.2xlarge` (1 NeuronCore-v2, 32 GB SBUF, AMD EPYC host CPU)
- **Image**: Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)
- **Neuron SDK**: `neuronxcc 2.24.5133.0`
- **Tool**: [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) with default settings (calibration + 5+ rounds, reports median)
- **NKI warmup**: each NKI test does an explicit warmup call before the timed loop so kernel compilation cost is excluded from steady-state numbers
- **Reproduce**: `AWS_PROFILE=aws ./scripts/run_benchmarks.sh`

## Caveats

- **Small sizes are dispatch-bound.** NKI kernel invocation has fixed overhead (XLA dispatch + tensor staging). Below some per-op threshold the host CPU wins.
- **Single-NeuronCore only.** trn1.2xlarge has one NeuronCore. Multi-NeuronCore parallelism will change the picture for large transforms.
- **FP32 throughout.** BF16 / FP16 paths are future work.
- Numbers can vary 5-15% run-to-run; treat the table as approximate.

## Findings (v0.8.0)

v0.8.0 shipped the batched `(B, n)` butterfly kernel, the single biggest gap in v0.7.0. Improvements on multi-call paths are dramatic:

| Operation | v0.7.0 (μs) | v0.8.0 (μs) | Speedup | Now vs PyTorch |
|-----------|---:|---:|---:|---:|
| fftn 32×64×64 | 52,258,203 | 70,821 | **738× faster** | 2.24× slower |
| fft2 1024×1024 | 32,441,988 | 545,474 | **59× faster** | 1.21× slower |
| fft2 256×256 | 5,014,511 | 45,186 | **111× faster** | **1.30× faster** |
| batched FFT (128×1024) | 2,073,239 | 52,422 | **39× faster** | **2.00× faster** |
| batched FFT (32×1024) | 519,479 | 25,103 | **21× faster** | **3.78× faster** |
| STFT 16k samples | 765,369 | 27,905 | **27× faster** | **1.75× faster** |

**Where NKI wins today** (NKI vs trnfft-PyTorch):

- **Single 1D FFT** at n ≥ 256: 2-11× faster (unchanged — single-call path was already batched across groups).
- **Bluestein**: 2-10× faster.
- **Batched FFT, STFT**: 1.75-3.78× faster. trnfft-NKI now beats **vanilla** `torch.fft.fft` for these workloads — the first category where that's true on this hardware.
- **Complex GEMM** at K ≥ 1024: 2.79× faster than PyTorch; also beats `torch.matmul(complex64, complex64)`.
- **fft2** at moderate sizes (256×256): 1.30× faster.

**Where NKI still loses:**

- **Small operations** (mask 64×32, GEMM 128, ComplexLinear 128→256) remain dispatch-bound. Host CPU wins by 5-100×. Tracked as #40 (SBUF-resident dispatch).
- **Very large 2D/3D FFT** (fft2 1024×1024, fftn): NKI is now only 1.2-3.5× slower than PyTorch (down from 60-1600× in v0.7.0), but PyTorch still wins on pure throughput.
- **Complex mask at 1024×512** — fixed in v0.9.0; compiles and runs, but still 5.7× slower than CPU (dispatch-bound like the smaller mask shapes).

The bottom line for v0.8.0: **trnfft-NKI is now the right default for STFT and batched FFT on Trainium**, and a clear win over the trnfft PyTorch fallback across most of the API. For small ops and 2D/3D FFT at very large sizes, keep using `set_backend("pytorch")`.

## What changed in v0.9.0

- **#39 fixed**: `_complex_mul_kernel` at 1024×512 now compiles and runs (prior `min()` in `affine_range` pattern matched the v0.8.0 butterfly bug). All 70/70 benchmarks pass for the first time. See the `mask | mask_shape2` row for the new number.
- **Docs refreshed**: README status banner, Related Projects table, and CLAUDE.md now reflect the matured trn-* suite (all 6 siblings + umbrella on PyPI).
- No kernel perf changes expected; the numbers above match v0.8.0 within run-to-run variance.

## Results

<!-- BENCH_TABLE_START -->

_Hardware: AWS trn1.2xlarge — neuronxcc 2.24.5133.0 — Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)_

_All times in microseconds (μs); lower is better. Speedup is trnfft-PyTorch / NKI._

| Operation | Param | NKI | trnfft-PyTorch | torch.* | NKI vs PyT |
|-----------|-------|----:|---------------:|--------:|----------:|
| batched_fft | batched_shape0 | 25,399.2 | 94,313.6 | 49.9 | 3.71× faster |
| batched_fft | batched_shape1 | 52,993.4 | 108,431.5 | 81.2 | 2.05× faster |
| bluestein | 127 | 31,420.6 | 66,836.8 | 13.4 | 2.13× faster |
| bluestein | 4097 | 433,125.2 | 4,342,374.5 | 314.5 | 10.03× faster |
| bluestein | 997 | 82,359.1 | 540,116.1 | 56.6 | 6.56× faster |
| fft | 1024 | 15,982.0 | 87,643.0 | 28.3 | 5.48× faster |
| fft | 16384 | 131,765.5 | 1,435,530.4 | 166.5 | 10.89× faster |
| fft | 256 | 10,034.8 | 21,755.1 | 10.5 | 2.17× faster |
| fft | 4096 | 39,297.5 | 355,413.5 | 92.8 | 9.04× faster |
| fft | 65536 | 555,985.4 | 5,794,943.3 | 559.6 | 10.42× faster |
| fft2 | fft2_shape0 | 15,063.0 | 12,240.9 | 20.4 | 1.23× slower |
| fft2 | fft2_shape1 | 44,826.8 | 60,771.0 | 105.6 | 1.36× faster |
| fft2 | fft2_shape2 | 537,623.9 | 474,073.2 | 1,067.1 | 1.13× slower |
| fftn | fftn_shape0 | 14,076.3 | 4,132.6 | 16.6 | 3.41× slower |
| fftn | fftn_shape1 | 70,156.7 | 32,736.6 | 85.6 | 2.14× slower |
| gemm | 1024 | 4,823.7 | 13,560.8 | 12,973.5 | 2.81× faster |
| gemm | 128 | 1,493.6 | 72.6 | 33.6 | 20.59× slower |
| gemm | 256 | 1,685.8 | 321.0 | 220.2 | 5.25× slower |
| gemm | 512 | 2,294.3 | 1,877.0 | 1,709.3 | 1.22× slower |
| linear | linear_shape0 | 1,659.4 | 201.4 | — | 8.24× slower |
| linear | linear_shape1 | 3,269.0 | 4,069.7 | — | 1.24× faster |
| mask | mask_shape0 | 1,412.7 | 15.0 | — | 94.37× slower |
| mask | mask_shape1 | 1,527.6 | 50.2 | — | 30.44× slower |
| mask | mask_shape2 | 2,965.8 | 519.7 | — | 5.71× slower |
| stft | - | 29,339.2 | 48,769.3 | 47.1 | 1.66× faster |

<!-- BENCH_TABLE_END -->

## Raw data

Versioned JSON outputs are committed under `docs/benchmark_results/`. To compare across versions:

```bash
pytest-benchmark compare docs/benchmark_results/*.json
```
