# trnfft

FFT and complex-valued tensor operations for AWS Trainium via NKI.

Trainium has no native complex number support and ships no FFT library. **trnfft** fills that gap with split real/imaginary representation, complex neural network layers, and NKI kernels optimized for the NeuronCore architecture.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Features

- **Drop-in `torch.fft` replacement** — `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `fftn`, `stft`, `istft`
- **ComplexTensor** — split real/imaginary representation with full arithmetic
- **Complex NN layers** — `ComplexLinear`, `ComplexConv1d`, `ComplexBatchNorm1d`, `ComplexModReLU`
- **NKI acceleration** — butterfly FFT, complex GEMM, and fused multiply kernels for Trainium
- **Plan-based caching** — FFTW-style plan creation and reuse

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
