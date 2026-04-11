# Architecture

## Layer diagram

```
+--------------------------------------------+
|            User Code / Model               |
+--------------------------------------------+
|         trnfft.api (torch.fft API)         |
|   fft()  ifft()  rfft()  stft()  fftn()   |
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
+------------------+------------------------+
```

## Key design decisions

1. **Split real/imaginary** — Trainium has no complex dtype. `ComplexTensor` wraps paired real tensors.

2. **Iterative Cooley-Tukey** — Radix-2 decimation-in-time with bit-reversal. Each butterfly stage vectorized across all groups.

3. **Bluestein for arbitrary sizes** — Converts arbitrary-N DFT to circular convolution via three power-of-2 FFTs.

4. **Plan-based execution** — Like FFTW/cuFFT. Plans cached by `(size, inverse)`.

5. **NKI dispatch** — Auto-detects Neuron hardware. Falls back to PyTorch on CPU/GPU.

## File structure

```
trnfft/
├── __init__.py          # Public API re-exports
├── api.py               # torch.fft-compatible functions
├── complex.py           # ComplexTensor
├── fft_core.py          # Cooley-Tukey + Bluestein
├── nn.py                # Complex NN layers
├── plan.py              # FFTPlan caching
└── nki/
    ├── dispatch.py      # Backend dispatch + GEMM/mul kernels
    ├── butterfly.py     # NKI butterfly kernel
    └── multicore.py     # Multi-NeuronCore (scaffold)
```
