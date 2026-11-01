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
├── precision.py         # set_precision / get_precision
└── nki/
    ├── dispatch.py      # Backend dispatch + GEMM/mul kernels
    ├── butterfly.py     # NKI butterfly kernel (fast + Kahan variants)
    ├── autograd.py      # torch.autograd.Function wrappers for kernels
    └── multicore.py     # Multi-NeuronCore batch split (host batches)
```

## Precision modes

Introduced in v0.11.0. Select via `trnfft.set_precision(mode)`. The table
covers the FP32 / FP64 modes; the BF16 modes (`"bf16"`, `"bf16_refined"`,
`"ozaki"`, `"ozaki_hq"`, N ≤ 256 DFT-GEMM) are documented in
`trnfft/precision.py`.

| Mode | Scope | Typical rel error (Bluestein) | Cost |
|------|-------|------------------------------|------|
| `"fast"` (default) | FP32 throughout | ~4e-7 at N = 1000 / 4097 (CPU) since the FP64 chirp fix in v0.23.0; it was 1.2e-4 at N = 1000 before | Baseline |
| `"kahan"` | Compensated Kahan/Dekker multiplies in Bluestein chirps; twoProd variant of NKI butterfly kernel | Matches "fast" on CPU (chirps aren't the error hotspot); meaningful reduction on NKI butterfly | ~2× butterfly op count on NKI |
| `"double"` | Promote Bluestein to FP64; power-of-2 N ≤ 1024 runs the DFT-GEMM on the host in FP64 | Near FP64 | No NKI: on the NKI path, FP64 work above the FP64 GEMM range computes on the host (v0.26.0, #109); power-of-2 N > 1024 runs the FP32 kernels |

**Recommendation:** use `"fast"` for STFT, batched FFT, and any
power-of-2 case. Use `"double"` when you need more than FP32 accuracy.
Since v0.23.0 (#82) `"kahan"` takes the same GEMM paths as `"fast"` up to
N = 4096 (they are more accurate on Trn2 than the Kahan butterfly) and is
never less accurate than `"fast"`.
