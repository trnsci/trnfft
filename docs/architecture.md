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
    └── multicore.py     # Multi-NeuronCore (scaffold)
```

## Precision modes

Introduced in v0.11.0. Select via `trnfft.set_precision(mode)`. The mode
mainly changes Bluestein's arbitrary-size FFT path; power-of-2 FFTs run
through Cooley-Tukey and are less error-prone.

| Mode | Scope | Typical rel error (Bluestein) | Cost |
|------|-------|------------------------------|------|
| `"fast"` (default) | FP32 throughout | ~2e-2 at N ∈ [500, 1000]; worse beyond | Baseline |
| `"kahan"` | Compensated Kahan/Dekker multiplies in Bluestein chirps; twoProd variant of NKI butterfly kernel | Matches "fast" on CPU (chirps aren't the error hotspot); meaningful reduction on NKI butterfly | ~2× butterfly op count on NKI |
| `"double"` | Promote entire Bluestein host math to FP64, cast back on exit | ~1e-11 at any N; 10+ orders of magnitude better than "fast" | Bluestein uses PyTorch FP64 (no NKI); power-of-2 FFTs unaffected |

**Recommendation:** use `"fast"` for STFT, batched FFT, and any
power-of-2 case. Use `"double"` when you need precision for Bluestein
sizes (non-power-of-2, N ≥ 500). `"kahan"` is useful only on Trainium
hardware where FP64 isn't available — on CPU it's equivalent to `"fast"`
for this particular API.

Source of the error: Bluestein runs three power-of-2 FFTs in series.
Each butterfly stage accumulates rounding error in the complex multiply
`prod_re = t_re*o_re - t_im*o_im`, and the three-FFT chain compounds
that loss. The chirp multiplications on the host are O(N) — compensating
them alone (as "kahan" on CPU does) doesn't reach the dominant O(N log N)
error source in the FFT itself.
