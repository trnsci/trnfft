# trnfft

FFT and complex-valued tensor operations for AWS Trainium via NKI.

Trainium has no native complex number support and ships no FFT library. **trnfft** fills that gap with split real/imaginary representation, complex neural network layers, and NKI kernels optimized for the NeuronCore architecture.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Features

- **`torch.fft`-compatible API** — `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft`, the `2` / `n` variants of each (`fft2`, `ifftn`, `hfftn`, ...), `fftshift`, `ifftshift`, `fftfreq`, `rfftfreq`, `stft`, `istft`. The transforms take `torch.fft`'s `n` / `s`, `dim` and `norm` arguments with the same meaning. `stft(window=None)` uses a Hann window, where `torch.stft` uses ones; pass `window` to match it.
- **ComplexTensor** — split real/imaginary representation with full arithmetic
- **Complex NN layers** — `ComplexLinear`, `ComplexConv1d`, `ComplexBatchNorm1d`, `ComplexModReLU`
- **NKI acceleration** — DFT-GEMM, four-step and six-step FFTs, Stockham and butterfly stages, complex GEMM, ComplexLinear, and fused multiply kernels for Trainium. Validated on Trn2 (Neuron SDK 2.33, native TorchNeuron) and on trn1.2xlarge (XLA). See [Benchmarks](benchmarks.md) and the CHANGELOG.
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
