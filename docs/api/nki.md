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
