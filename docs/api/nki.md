# NKI Backend

trnfft auto-detects Trainium hardware and dispatches to optimized NKI kernels when available.

## Backend control

```python
import trnfft

trnfft.HAS_NKI  # True if nki (>=0.3.0) is installed
trnfft.get_backend()  # Current setting: "auto", "pytorch", or "nki"
trnfft.set_backend("auto")  # NKI if available, else PyTorch (default)
trnfft.set_backend("pytorch")  # Force PyTorch (any device)
trnfft.set_backend("nki")  # Force NKI (fails if not on Trainium)
```

## Environment variables

| Variable | Effect |
| --- | --- |
| `TRNFFT_USE_SIMULATOR=1` | Route NKI kernels through `nki.simulate(kernel)(np_args)` on CPU instead of the NEFF + hardware path. Requires `nki>=0.3.0`. Correctness iteration only; simulator explicitly skips NEFF compile and SBUF/PSUM capacity checks. See [developing_kernels.md](../developing_kernels.md). |
| `TRNFFT_SIX_STEP=0\|1` | The six-step FFT is the default for power-of-2 FP32 N ≥ 8192 (since v0.25.0, #94). `0` restores the Stockham (`"fast"`) / compensated butterfly (`"kahan"`) path; `1` forces the six-step for every N above 4096. `TRNFFT_SIX_STEP_MIN` moves the threshold. |
| `TRNFFT_PE_TRANSPOSE=0` | Selects the older kernels that read their input with `nl.load_transpose2d` (the XLA path keeps them by default). Since v0.25.0 the native path transposes on the Tensor engine instead, because the runtime rebuilt a `load_transpose2d` input's DMA rings at every new input address (#97, #100). |
| `TRNFFT_PINNED_OUTPUT=0` | Turns off pinned-memory host outputs (planes of 1 MiB and up, native path; since v0.25.0, #98). |
| `TRNFFT_REQUIRE_NKI=1` | Refuse operand dtypes the kernels do not take (`dispatch.KERNEL_DTYPES`: float32/bfloat16/float16, float32 only for the fused four-step and the six-step GEMM). These raise `NotImplementedError` instead of computing float64 in torch (`auto`) or in float32 on the kernel (`nki`), or computing integer, bool and complex operands in torch. `precision="double"` keeps its host FP64 route. Read per call. See [Precision and dtypes](#precision-and-dtypes). |
| `TRNFFT_FP64_ON_DEVICE=1` | Under `auto`, keep float64 / complex128 operands that are already on the neuron device on device aten (FP32 precision, no host copy, no warning) instead of computing them on the host at float64 accuracy. Read per call. See [Precision and dtypes](#precision-and-dtypes). |
| `TRNSCI_NKI_DISPATCH=auto\|native\|xla\|cpu` | Suite-wide. Forces the device dispatch path (see below); `auto` (default) detects. Forcing an unavailable path raises. |
| `TRNSCI_XLA_DEPRECATION=1` | Emit one `DeprecationWarning` when the XLA path is selected. Off by default. |

## Precision and dtypes

Trainium computes float64 at FP32 precision, so no NKI kernel takes float64
(or complex128, whose parts are float64). trnfft follows the trnsci suite
policy (#136) for float64 operands on the NKI path, in every transform,
`stft`/`istft`, `complex_gemm`, `complex_linear` / `ComplexLinear`,
`complex_mask_apply` and `complex_gemm_bf16`:

| Setting | float64 behaviour |
| --- | --- |
| `set_backend("auto")` (default) | Computes in host torch at float64 accuracy. No kernel runs. Host-resident operands compute where they are, as `set_backend("pytorch")` does. Operands already on the neuron device are copied to the host, computed there, and the result is returned on their device in float64 / complex128 (see below). |
| `set_backend("nki")` | The kernel runs on float32 copies; each op warns once (`UserWarning`) that precision is reduced to float32. |
| `TRNFFT_REQUIRE_NKI=1` | Raises `NotImplementedError`, on either backend. |
| `precision="double"` | Keeps its host FP64 route on either backend (no warning, no raise). |

**float64 already on the neuron device.** Device aten on Trainium also
computes float64 at FP32 precision (measured on Trn2), so under `auto` the
torch route for a float64 or complex128 operand that is already on the neuron
device runs on the host: every transform (`fft` … `ihfftn`, `stft`, `istft`),
`complex_gemm`, `complex_linear`, `complex_mask_apply`, `ComplexLinear` and
`ComplexConv1d` copy their neuron-resident operands to the host, compute in float64 and return
the result on the original device in float64 / complex128. The copies are
differentiable. Each op emits one `UserWarning`: "float64 on the neuron device
is computed on the host for float64 accuracy (device arithmetic is fp32); pass
float32 to stay on device". Set `TRNFFT_FP64_ON_DEVICE=1` to keep these on
device aten instead (faster, FP32 precision, no warning). `set_backend("nki")`,
`set_backend("pytorch")` and `TRNFFT_REQUIRE_NKI=1` are unaffected.
`complex_gemm_bf16` (FP32 by design), `complex_matmul`, `ComplexBatchNorm1d`
and `ComplexModReLU` are plain torch on the operands' device.

`ComplexConv1d` has no kernel, and a float64 `aten::convolution` does not
compile on the neuron device (#134). So float64 that stays on the device
(`set_backend("nki")`, or `TRNFFT_FP64_ON_DEVICE=1`) is convolved in float32
there and returned as float64; `"nki"` warns once. `TRNFFT_REQUIRE_NKI=1`
raises `NotImplementedError` for it, and `set_backend("pytorch")` runs plain
torch (the compile error).

The result dtype is always what the pytorch backend returns for the same
inputs: float64 in gives float64 out; bfloat16/float16 in gives the input
dtype, also in the low-precision modes (`bf16`, `bf16_refined`, `ozaki`,
`ozaki_hq`), whose BF16 GEMM accumulates in an FP32 PSUM. Mixed operands that
include float64 give `torch.promote_types` of the operands. bfloat16 and
float16 operands are passed to the kernels as they are (#137 tracks whether to
upcast them). `complex_gemm_bf16` returns float32 on every backend. Integer,
bool and complex `torch.Tensor` operands of the dispatch-level ops compute in
torch; the transforms promote integer and bool input first, as `torch.fft`
does.

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
