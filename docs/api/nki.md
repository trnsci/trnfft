# NKI Backend

trnfft auto-detects Trainium hardware and dispatches to optimized NKI kernels when available.

## Backend control

```python
import trnfft

trnfft.HAS_NKI          # True if nki (>=0.3.0) is installed
trnfft.get_backend()    # Current setting: "auto", "pytorch", or "nki"
trnfft.set_backend("auto")     # NKI if available, else PyTorch (default)
trnfft.set_backend("pytorch")  # Force PyTorch (any device)
trnfft.set_backend("nki")      # Force NKI (fails if not on Trainium)
```

## Environment variables

| Variable | Effect |
| --- | --- |
| `TRNFFT_USE_SIMULATOR=1` | Route NKI kernels through `nki.simulate(kernel)(np_args)` on CPU instead of the NEFF + hardware path. Requires `nki>=0.3.0`. Correctness iteration only; simulator explicitly skips NEFF compile and SBUF/PSUM capacity checks. See [developing_kernels.md](../developing_kernels.md). |

| `TRNSCI_NKI_DISPATCH=auto\|native\|xla\|cpu` | Suite-wide. Forces the device dispatch path (see below); `auto` (default) detects. Forcing an unavailable path raises. |
| `TRNSCI_XLA_DEPRECATION=1` | Emit one `DeprecationWarning` when the XLA path is selected. Off by default. |

## Dispatch path

On Neuron hardware, kernels run on one of two `torch-neuronx` lineages,
detected on the first kernel launch:

- **Native (TorchNeuron, Trn2/Trn3)**: each kernel is a PyTorch custom op on
  `device="neuron"`, also callable directly and inside
  `torch.compile(backend="neuron")`: `torch.ops.trnfft.complex_gemm`,
  `complex_gemm_bf16`, `complex_linear`, `complex_mul`, `butterfly_stage`,
  `butterfly_stage_kahan`, `stockham_radix4_stage`, `stockham_radix8_w8`.
  Each takes and returns real/imag tensor pairs. `torch.ops.trnfft.fft`
  (`x_real, x_imag, inverse, precision`) is the composite FFT that carries
  the analytic adjoint; `complex_gemm`, `complex_linear` and `complex_mul`
  have theirs attached with `torch.library.register_autograd`. The ops are
  registered only once this path is selected.
- **XLA (Trn1)**: kernels run on the `torch_xla` device, as before.

`set_backend` is unaffected: it still chooses NKI vs PyTorch. See
[Installation](../installation.md#two-neuron-pytorch-lineages).

## NKI kernels

### Complex GEMM

Stationary tile reuse on the Tensor Engine systolic array:

- Phase 1: A\_real stationary, stream B\_real and B\_imag
- Phase 2: A\_imag stationary, stream -B\_imag and B\_real
- 4 SBUF loads instead of 8 (50% fewer HBM transfers)

### Fused complex multiply

Element-wise `(a+bi)(c+di)` in a single kernel invocation. Loads all 4 inputs in one pass, computes `ac-bd` and `ad+bc` in SBUF, writes 2 outputs. Replaces 6 separate HBM round-trips.

### Butterfly FFT

Each Cooley-Tukey butterfly stage dispatches to an NKI kernel that processes all butterflies using the Vector Engine. Twiddle factors are preloaded to SBUF and reused across the batch dimension.

## Architecture

```
+------------------+------------------------+
|  PyTorch ops     |  NKI kernels           |
|  (any device)    |  (Trainium only)       |
|  torch.matmul    |  nisa.nc_matmul        |
|  element-wise    |  Tensor Engine         |
|                  |  Vector Engine          |
|                  |  SBUF <-> PSUM pipeline |
+------------------+------------------------+
```
