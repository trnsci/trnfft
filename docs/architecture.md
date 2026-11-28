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
| `"dw"` | Power-of-2 N in [2, 4096]: double-word (float-float) FFT, every value hi + lo in FP32, butterflies from error-free FP32 transformations (#172); other N take the `"double"` route | 47.5–48.9 correct bits (CPU, against mpmath; host FP64 51.9–53.0, FP32 22.8–24.0) | One NKI launch on the Vector engine (optionally Vector + GpSimd): 164 FP32 ops per point per twiddled radix-4 stage (115 with `add="sloppy"`), 900 per point at N = 4096; the CPU reference gives the same bits |

**Recommendation:** use `"fast"` for STFT, batched FFT, and any
power-of-2 case. Use `"double"` when you need more than FP32 accuracy.
`"dw"` is the device arm of fp64-class accuracy (#172): its device timing
against the host FP64 route is measured in the hardware window, so
`"double"` stays the default accuracy-first mode until then.

### Double-word FFT (`"dw"`, `trnfft/dw.py`)

A Stockham autosort FFT, radix 4 with one leading radix-2 stage when
log2 N is odd, on double-word complex data (four FP32 planes). The
building blocks are Knuth's TwoSum, Dekker's FastTwoSum and FMA-free TwoProd
(Veltkamp split), and the double-word addition and multiplication of
Joldes, Muller and Popescu (ACM TOMS 44(2), 2017); complex products are a sum
of two double-word products per component (Stloukal and Jézéquel,
arXiv:2611.08371). Twiddles are computed in FP64 on the host with an octant
reduction and split exactly into hi + lo.

Why this maps to Trainium: the Vector and GpSimd engines round FP32 add,
subtract and multiply to nearest even, so the error-free transformations
are exact there (the E0 probe, trnsci/trnsci#169). The Tensor engine is not
used: its FP32 products are not correctly rounded. There is no Vector-engine
FMA, and the Scalar engine's fused `activation(scale * x + bias)` needs a
per-partition factor (the twiddles vary along the free axis here) and
flushes subnormal products, so every step is one unfused instruction.

The kernel (`trnfft/nki/dw_fft.py`) runs the whole transform in one
launch with the data in SBUF: each stage's output layout is the next stage's
input layout, so there is no permutation. It runs the reference's own
butterfly code through an ops object that issues one instruction per FP32
operation, so a device result equals the CPU reference bit for bit at any
batch, tiling and LNC (checked under `nki.simulate` at LNC = 1 and 2).

Since v0.23.0 (#82) `"kahan"` takes the same GEMM paths as `"fast"` up to
N = 4096 (they are more accurate on Trn2 than the Kahan butterfly) and is
never less accurate than `"fast"`.
