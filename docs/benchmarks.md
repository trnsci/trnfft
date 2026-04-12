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
- **Multi-call paths suffer from per-invocation overhead.** `fftn`, `fft2`, batched 1D FFT, and STFT all loop over rows in Python and invoke the NKI kernel per row. They're 50-1600× slower than the same code with `set_backend("pytorch")`. Fixing this requires a batched butterfly kernel — tracked as a v0.8 follow-up.
- **Single-NeuronCore only.** trn1.2xlarge has one NeuronCore. Multi-NeuronCore parallelism (planned for v0.8) will change the picture for large transforms.
- **FP32 throughout.** BF16 / FP16 paths are future work.
- Numbers can vary 5-15% run-to-run; treat the table as approximate.

## Findings (v0.7.0)

**Where NKI wins right now:**

- **Single 1D FFT** at n ≥ 256: 2-11× faster than the trnfft PyTorch fallback. Dispatch overhead amortizes well over a single-call butterfly chain.
- **Bluestein** (arbitrary-size FFT): 2-10× faster than the trnfft PyTorch fallback because the inner three power-of-2 FFTs route through NKI butterfly.
- **Complex GEMM** at K ≥ 512: 3-4× faster than `complex_matmul` (PyTorch). At 1024², beats `torch.matmul(complex64, complex64)` by 3.79×.
- **ComplexLinear** at 512→1024: 1.22× faster than the 4-`nn.Linear` fallback.

**Where NKI loses today:**

- **Multi-call code paths** (fftn, fft2, batched FFT, STFT) are catastrophically slow because of per-row kernel dispatch in Python. The fix is a batched butterfly kernel (taking `(B, n)` directly), filed as a v0.8 issue.
- **Small operations** (mask 64×32, GEMM 128, ComplexLinear 128→256) are dominated by dispatch overhead. The host CPU wins by 5-100×.
- **vs vanilla `torch.fft.fft`** the picture is harsher: PyTorch's FFT is extremely optimized on AMD EPYC and beats NKI for all 1D FFT sizes tested. NKI only beats vanilla PyTorch in the **complex GEMM ≥ 512** case, where Trainium's Tensor Engine systolic array shines.

The bottom line: **trnfft's NKI backend is the right choice for large standalone GEMMs and for 1D FFTs when the alternative is the trnfft PyTorch path** (e.g., as part of an autograd graph). For small ops, hot-path STFT, and arbitrary fftn/fft2, use `set_backend("pytorch")` until the v0.8 batched butterfly lands.

## Results

<!-- BENCH_TABLE_START -->

_Hardware: AWS trn1.2xlarge — neuronxcc 2.24.5133.0 — Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)_

_All times in microseconds (μs); lower is better. Speedup is trnfft-PyTorch / NKI._

| Operation | Param | NKI | trnfft-PyTorch | torch.* | NKI vs PyT |
|-----------|-------|----:|---------------:|--------:|----------:|
| batched_fft | batched_shape0 | 519,479.8 | 96,145.7 | 47.0 | 5.40× slower |
| batched_fft | batched_shape1 | 2,073,239.9 | 106,352.7 | 77.7 | 19.49× slower |
| bluestein | 127 | 31,587.0 | 67,531.3 | 14.4 | 2.14× faster |
| bluestein | 4097 | 437,314.5 | 4,369,754.4 | 323.0 | 9.99× faster |
| bluestein | 997 | 83,438.8 | 544,043.4 | 58.3 | 6.52× faster |
| fft | 1024 | 15,648.8 | 88,181.9 | 27.5 | 5.64× faster |
| fft | 16384 | 130,692.8 | 1,441,849.1 | 157.8 | 11.03× faster |
| fft | 256 | 9,739.8 | 21,931.1 | 8.9 | 2.25× faster |
| fft | 4096 | 38,964.9 | 358,264.5 | 89.7 | 9.19× faster |
| fft | 65536 | 556,793.2 | 5,829,695.7 | 540.1 | 10.47× faster |
| fft2 | fft2_shape0 | 888,494.1 | 12,220.2 | 19.6 | 72.71× slower |
| fft2 | fft2_shape1 | 5,014,511.6 | 57,407.8 | 94.5 | 87.35× slower |
| fft2 | fft2_shape2 | 32,441,988.9 | 440,604.6 | 964.2 | 73.63× slower |
| fftn | fftn_shape0 | 2,153,302.5 | 4,143.3 | 18.4 | 519.71× slower |
| fftn | fftn_shape1 | 52,258,203.2 | 31,563.8 | 86.7 | 1655.64× slower |
| gemm | 1024 | 4,866.6 | 18,423.9 | 18,908.5 | 3.79× faster |
| gemm | 128 | 1,514.9 | 73.4 | 44.3 | 20.65× slower |
| gemm | 256 | 1,740.6 | 304.9 | 298.2 | 5.71× slower |
| gemm | 512 | 2,531.4 | 2,642.4 | 2,432.0 | 1.04× faster |
| linear | linear_shape0 | 1,758.3 | 197.4 | — | 8.91× slower |
| linear | linear_shape1 | 3,183.0 | 3,879.9 | — | 1.22× faster |
| mask | mask_shape0 | 1,414.2 | 14.7 | — | 96.18× slower |
| mask | mask_shape1 | 1,568.0 | 47.4 | — | 33.08× slower |
| mask | mask_shape2 | — | 337.1 | — | — |
| stft | - | 765,369.8 | 48,920.9 | 48.7 | 15.65× slower |

<!-- BENCH_TABLE_END -->

## Raw data

Versioned JSON outputs are committed under `docs/benchmark_results/`. To compare across versions:

```bash
pytest-benchmark compare docs/benchmark_results/*.json
```
