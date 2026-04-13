# trnfft

[![CI](https://github.com/trnsci/trnfft/actions/workflows/ci.yml/badge.svg)](https://github.com/trnsci/trnfft/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trnfft)](https://pypi.org/project/trnfft/)
[![Python](https://img.shields.io/pypi/pyversions/trnfft)](https://pypi.org/project/trnfft/)
[![License](https://img.shields.io/github/license/trnsci/trnfft)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://trnsci.github.io/trnfft/)

FFT and complex-valued tensor operations for AWS Trainium via NKI.

Trainium has no native complex number support and ships no FFT library. `trnfft` fills that gap with split real/imaginary representation, complex neural network layers, and NKI kernels optimized for the NeuronCore architecture.

Incorporates [neuron-complex-ops](https://github.com/scttfrdmn/neuron-complex-ops). Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Why

NVIDIA has cuFFT, cuBLAS, and native `complex64`. Trainium has none of these. Every signal processing, speech enhancement, physics simulation, and spectral method workload on Trainium currently falls back to CPU or requires hand-rolling complex arithmetic. trnfft fixes this.

## Install

```bash
pip install trnfft

# With Neuron hardware support
pip install trnfft[neuron]
```

## Usage

```python
import torch
import trnfft

# Drop-in replacement for torch.fft
signal = torch.randn(1024)
X = trnfft.fft(signal)
recovered = trnfft.ifft(X)

# Real-valued FFT
X = trnfft.rfft(signal)

# 2D FFT
image = torch.randn(256, 256)
F = trnfft.fft2(image)

# STFT (matches torch.stft signature)
waveform = torch.randn(16000)
S = trnfft.stft(waveform, n_fft=512, hop_length=256)
```

## Complex Neural Network Layers

```python
from trnfft import ComplexTensor
from trnfft.nn import ComplexLinear, ComplexConv1d, ComplexModReLU

# Build complex-valued models for speech/audio/physics
x = ComplexTensor(real_part, imag_part)
layer = ComplexLinear(256, 128)
y = layer(x)
```

## Architecture

```
+--------------------------------------------+
|            User Code / Model               |
+--------------------------------------------+
|         trnfft.api (torch.fft API)         |
|   fft()  ifft()  rfft()  stft()  fft2()   |
+--------------------------------------------+
|   trnfft.fft_core     |  trnfft.nn        |
|   Cooley-Tukey         |  ComplexLinear    |
|   Bluestein            |  ComplexConv1d    |
|   Plan caching         |  ComplexModReLU   |
+------------------------+-------------------+
|       trnfft.nki.dispatch                  |
|   "auto" | "pytorch" | "nki"              |
+--------------------------------------------+
|  PyTorch ops     |  NKI kernels           |
|  (any device)    |  (Trainium only)       |
|  torch.matmul    |  nisa.nc_matmul        |
|  element-wise    |  Tensor Engine         |
|                  |  Vector Engine          |
|                  |  SBUF ↔ PSUM pipeline  |
+------------------+------------------------+
```

## How It Works

**No complex dtype?** Trainium's NKI doesn't support `complex64`/`complex128`. `ComplexTensor` stores complex values as paired real tensors and decomposes complex arithmetic into real-valued operations.

**FFT → butterflies → matmul.** Each Cooley-Tukey butterfly stage performs complex-multiply-and-add across all groups simultaneously. On NKI, the complex multiply maps to the Tensor Engine (systolic array).

**Algorithms:**
- **Power-of-2**: Cooley-Tukey radix-2 (iterative, decimation-in-time)
- **Arbitrary sizes**: Bluestein's chirp-z transform (pads to power-of-2)

**NKI complex GEMM** uses stationary tile reuse (2 SBUF loads instead of 8) and PSUM accumulation, overlapping Vector Engine negation with Tensor Engine matmul.

## Hardware compatibility

NKI kernels are validated against **Neuron SDK 2.24+** on the **Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)** AMI (20260410 or later). See [docs/installation.md](https://trnsci.github.io/trnfft/installation/#hardware-compatibility) for the full compatibility matrix.

## Benchmarks

NKI vs PyTorch on the same Trainium instance — see the [benchmarks page](https://trnsci.github.io/trnfft/benchmarks/) for the latest numbers.

## Status

**v0.8.0** — NKI butterfly, GEMM, complex-multiply, and ComplexLinear kernels are validated on trn1.2xlarge. For STFT and batched FFT, `set_backend("nki")` now beats vanilla `torch.fft.fft`. See [benchmarks](https://trnsci.github.io/trnfft/benchmarks/) for the full picture.

**API coverage** (9 of 12 `torch.fft` functions):
`fft`, `ifft`, `rfft`, `irfft`, `fft2`, `fftn`, `ifftn`, `stft`, `istft`.
Not yet: `hfft`, `ihfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn` (tracked for v0.10.0+).

**Roadmap**
- NKI `ComplexConv1d` / `ComplexModReLU` kernels (today both fall back to PyTorch on NKI)
- BF16 / FP16 support across NKI kernels
- Multi-NeuronCore parallelism (scaffold in `trnfft/nki/multicore.py`)
- SBUF-resident dispatch to reduce small-op overhead
- Remaining `torch.fft` functions

## Related projects in the trnsci suite

All six siblings are on PyPI, along with the umbrella meta-package:

| Project | What | Latest |
|---------|------|-------:|
| [trnsci](https://github.com/trnsci/trnsci) | Umbrella meta-package pulling the whole suite | v0.1.0 |
| [trnblas](https://github.com/trnsci/trnblas) | BLAS Level 1–3 for Trainium | v0.4.0 |
| [trnrand](https://github.com/trnsci/trnrand) | Philox / Sobol / Halton random number generation | v0.1.0 |
| [trnsolver](https://github.com/trnsci/trnsolver) | Linear solvers (CG, GMRES) and eigendecomposition | v0.3.0 |
| [trnsparse](https://github.com/trnsci/trnsparse) | Sparse matrix operations | v0.1.1 |
| [trntensor](https://github.com/trnsci/trntensor) | Tensor contractions (einsum, TT/Tucker decompositions) | v0.1.1 |
| [neuron-complex-ops](https://github.com/scttfrdmn/neuron-complex-ops) | Original proof-of-concept, folded into trnfft | archived |

## License

Apache 2.0 — Copyright 2026 Scott Friedman

## Acknowledgments

Built on insights from:
- [tcFFT](https://github.com/tcFFT) — Tensor Core FFT research
- [FFTW](https://www.fftw.org/) — Plan-based FFT architecture
- [AWS NKI documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
