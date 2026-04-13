# trnfft

FFT and complex-valued tensor operations for AWS Trainium via NKI.

Trainium has no native complex number support and ships no FFT library. **trnfft** fills that gap with split real/imaginary representation, complex neural network layers, and NKI kernels optimized for the NeuronCore architecture.

Part of the trnsci scientific computing suite ([github.com/trnsci](https://github.com/trnsci)).

## Features

- **`torch.fft`-compatible API** — `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn`, `stft`, `istft` (13 of ~15; `hfft` / `ihfft` are not implemented — see the [API stance](#hfft-ihfft-not-implemented) below)
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

## `hfft` / `ihfft` — not implemented

The two `torch.fft` functions trnfft doesn't provide are `hfft` (Hermitian
input → real output) and its inverse `ihfft`. These expect a
conjugate-symmetric input tensor `X[k] = conj(X[N-k])`, which in practice
only arises if you've just produced one via `rfft` — at which point the
natural continuation is `irfft`, not `hfft`.

**When you'd want them:** if your workload *directly* produces a
Hermitian-symmetric spectrum (e.g., reconstructing a real signal from a
known symmetric frequency-domain representation) and you don't want the
manual unpack/pack step that gets you there via `rfft` / `irfft`.

**Workaround today:** pack your symmetric input into the first `N//2+1`
bins and call `irfft`. Unpack an `rfft` output to the full N bins when a
Hermitian-input consumer expects it.

**If you need these:** [open an issue](https://github.com/trnsci/trnfft/issues)
with the concrete workload — the NKI butterfly kernels already implement
the primitives; it's a matter of adding the normalization + axis
conventions that match PyTorch.

## License

Apache 2.0 — Copyright 2026 Scott Friedman
