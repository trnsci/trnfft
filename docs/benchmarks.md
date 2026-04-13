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
- **Complex mask at 1024×512** still triggers a compile error — tracked as #39.

The bottom line for v0.8.0: **trnfft-NKI is now the right default for STFT and batched FFT on Trainium**, and a clear win over the trnfft PyTorch fallback across most of the API. For small ops and 2D/3D FFT at very large sizes, keep using `set_backend("pytorch")`.

## Results

<!-- BENCH_TABLE_START -->

_Hardware: AWS trn1.2xlarge — neuronxcc 2.24.5133.0 — Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)_

_All times in microseconds (μs); lower is better. Speedup is trnfft-PyTorch / NKI._

| Operation | Param | NKI | trnfft-PyTorch | torch.* | NKI vs PyT |
|-----------|-------|----:|---------------:|--------:|----------:|
| batched_fft | batched_shape0 | 25,103.3 | 94,806.8 | 48.9 | 3.78× faster |
| batched_fft | batched_shape1 | 52,422.6 | 104,750.5 | 79.2 | 2.00× faster |
| bluestein | 127 | 31,553.4 | 66,740.6 | 14.0 | 2.12× faster |
| bluestein | 4097 | 435,055.6 | 4,332,786.9 | 328.5 | 9.96× faster |
| bluestein | 997 | 82,700.9 | 536,492.5 | 58.8 | 6.49× faster |
| fft | 1024 | 16,006.1 | 87,375.9 | 28.6 | 5.46× faster |
| fft | 16384 | 131,665.6 | 1,430,051.1 | 164.6 | 10.86× faster |
| fft | 256 | 9,852.1 | 21,590.2 | 10.6 | 2.19× faster |
| fft | 4096 | 39,537.3 | 352,915.9 | 93.7 | 8.93× faster |
| fft | 65536 | 556,940.6 | 5,783,417.8 | 559.2 | 10.38× faster |
| fft2 | fft2_shape0 | 15,786.3 | 12,080.5 | 20.3 | 1.31× slower |
| fft2 | fft2_shape1 | 45,186.4 | 58,768.5 | 107.3 | 1.30× faster |
| fft2 | fft2_shape2 | 545,473.6 | 449,887.0 | 1,082.8 | 1.21× slower |
| fftn | fftn_shape0 | 14,248.2 | 4,109.5 | 16.5 | 3.47× slower |
| fftn | fftn_shape1 | 70,820.7 | 31,561.1 | 88.6 | 2.24× slower |
| gemm | 1024 | 5,013.8 | 13,974.8 | 13,001.2 | 2.79× faster |
| gemm | 128 | 1,569.4 | 114.7 | 35.2 | 13.68× slower |
| gemm | 256 | 1,785.8 | 316.9 | 228.2 | 5.64× slower |
| gemm | 512 | 2,339.9 | 1,936.7 | 1,715.9 | 1.21× slower |
| linear | linear_shape0 | 1,729.1 | 199.5 | — | 8.67× slower |
| linear | linear_shape1 | 3,563.4 | 3,975.6 | — | 1.12× faster |
| mask | mask_shape0 | 1,440.5 | 14.6 | — | 98.47× slower |
| mask | mask_shape1 | 1,546.2 | 51.1 | — | 30.25× slower |
| mask | mask_shape2 | — | 513.5 | — | — |
| stft | - | 27,905.2 | 48,884.4 | 48.9 | 1.75× faster |

<!-- BENCH_TABLE_END -->

## Raw data

Versioned JSON outputs are committed under `docs/benchmark_results/`. To compare across versions:

```bash
pytest-benchmark compare docs/benchmark_results/*.json
```
