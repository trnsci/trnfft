# trnfft

FFT and complex-valued tensor operations for AWS Trainium via NKI.

Trainium has no native complex number support and ships no FFT library. **trnfft** fills that gap with split real/imaginary representation, complex neural network layers, and NKI kernels optimized for the NeuronCore architecture.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Features

- **`torch.fft`-compatible API** — `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn`, `stft`, `istft` (13 of ~15; `hfft` and `ihfft` — Hermitian-input variants — are the only transforms not implemented)
- **ComplexTensor** — split real/imaginary representation with full arithmetic
- **Complex NN layers** — `ComplexLinear`, `ComplexConv1d`, `ComplexBatchNorm1d`, `ComplexModReLU`
- **NKI acceleration** — butterfly FFT, complex GEMM, ComplexLinear, and fused multiply kernels for Trainium. Validated on trn1.2xlarge; beats vanilla `torch.fft` for STFT and batched FFT. See [Benchmarks](benchmarks.md).
- **Plan-based caching** — FFTW-style plan creation and reuse

## trn-* suite

trnfft is one of six packages (plus an umbrella meta-package) in the trnsci scientific computing suite for AWS Trainium:

- [trnblas](https://github.com/trnsci/trnblas) — BLAS Level 1–3
- [trnrand](https://github.com/trnsci/trnrand) — Philox / Sobol RNG
- [trnsolver](https://github.com/trnsci/trnsolver) — linear solvers, eigendecomposition
- [trnsparse](https://github.com/trnsci/trnsparse) — sparse matrix ops
- [trntensor](https://github.com/trnsci/trntensor) — tensor contractions
- [trnsci](https://github.com/trnsci/trnsci) — umbrella meta-package

## Quick example

```python
import torch
import trnfft

signal = torch.randn(1024)
X = trnfft.fft(signal)
recovered = trnfft.ifft(X)
```

## License

Apache 2.0 — Copyright 2026 Scott Friedman
