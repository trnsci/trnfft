# trnfft

[![CI](https://github.com/scttfrdmn/trnfft/actions/workflows/ci.yml/badge.svg)](https://github.com/scttfrdmn/trnfft/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trnfft)](https://pypi.org/project/trnfft/)
[![Python](https://img.shields.io/pypi/pyversions/trnfft)](https://pypi.org/project/trnfft/)
[![License](https://img.shields.io/github/license/scttfrdmn/trnfft)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://scttfrdmn.github.io/trnfft/)

FFT and complex-valued tensor operations for AWS Trainium via NKI.

Trainium has no native complex number support and ships no FFT library. `trnfft` fills that gap with split real/imaginary representation, complex neural network layers, and NKI kernels optimized for the NeuronCore architecture.

Incorporates [neuron-complex-ops](https://github.com/scttfrdmn/neuron-complex-ops). Part of the **trn-\*** scientific computing suite by [Playground Logic](https://playgroundlogic.co).

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

## Status

**v0.1.0** — CPU fallback works, NKI kernels scaffolded for on-hardware validation.

- [x] ComplexTensor with full arithmetic
- [x] Complex matmul (4 real matmuls)
- [x] 1D FFT/IFFT (power-of-2, Cooley-Tukey)
- [x] Bluestein (arbitrary sizes)
- [x] rfft/irfft
- [x] 2D FFT
- [x] STFT
- [x] Complex NN layers (Linear, Conv1d, BatchNorm, ModReLU)
- [x] NKI dispatch layer (auto/pytorch/nki)
- [x] Plan caching
- [ ] NKI butterfly kernel validation on trn1/trn2
- [ ] NKI GEMM kernel validation
- [ ] Multi-NeuronCore parallelism
- [ ] Benchmarks vs cuFFT
- [ ] Inverse STFT
- [ ] N-D FFT

## Related Projects

| Project | What |
|---------|------|
| [neuron-complex-ops](https://github.com/scttfrdmn/neuron-complex-ops) | Original proof-of-concept (now folded into this library) |
| trnblas *(planned)* | BLAS for Trainium |
| trnsolver *(planned)* | Linear solvers and eigendecomposition for Trainium |

## License

Apache 2.0 — Playground Logic LLC

## Acknowledgments

Built on insights from:
- [tcFFT](https://github.com/tcFFT) — Tensor Core FFT research
- [FFTW](https://www.fftw.org/) — Plan-based FFT architecture
- [AWS NKI documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
