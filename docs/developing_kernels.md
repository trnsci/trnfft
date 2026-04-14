# Developing NKI kernels for trnfft

trnfft's kernel dev loop mirrors the suite-wide pattern documented at
[trnsci/docs/developing_kernels.md](https://github.com/trnsci/trnsci/blob/main/docs/developing_kernels.md).
This page captures the trnfft-specific pieces.

## Two-tier dispatch

Kernels dispatch through one of two paths, selected by environment:

| Mode | Trigger | Runs on | Catches |
| --- | --- | --- | --- |
| Hardware | `set_backend("nki")` + trn1/trn2 + AMI venv | Trainium | Everything (compile, MLIR, perf, numerics) |
| Simulator | `TRNFFT_USE_SIMULATOR=1` + `nki>=0.3.0` on any Linux host | CPU | Python-trace-level errors (bad kwargs, dropped ops, shape mismatches) |

Simulator explicitly skips NEFF compile, so MLIR verifier errors and
SBUF/PSUM capacity issues remain hardware-only. It's a correctness pre-check,
not a replacement for hardware validation.

## Running the two paths

```bash
# Hardware (requires AWS_PROFILE; starts/stops trn1 CI instance)
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh

# Simulator on the CI instance
AWS_PROFILE=aws ./scripts/run_simulator_tests.sh

# Simulator in GH Actions (no AWS needed)
# runs automatically on push via the `nki-simulator` job
```

## Dispatch-code pattern

The autograd `forward` and the FFT driver branch on `_use_simulator()`:

```python
if _use_simulator():
    c_real, c_imag = _simulate_kernel(
        _complex_gemm_kernel, a_real, a_imag, b_real, b_imag
    )
else:
    (ar, ai, br, bi), orig = _to_xla(a_real, a_imag, b_real, b_imag)
    c_real, c_imag = _complex_gemm_kernel(ar, ai, br, bi)
    c_real, c_imag = c_real.to(orig), c_imag.to(orig)
```

`_simulate_kernel(kernel, *torch_tensors)` (in `trnfft/nki/dispatch.py`)
marshals numpy in/out, calling `nki.simulate(kernel)(*np_args)`.

## NKI 0.3.0 calling convention (since this is the floor)

- `nisa.nc_matmul(dst=, stationary=, moving=, accumulate=True)` — all kwargs,
  in-place accumulation into the PSUM tile.
- `nisa.tensor_copy` to materialize PSUM→SBUF. `nl.copy` now returns a view.
- Tensor-tensor `nl.divide` is gone; use `nl.multiply(x, nl.reciprocal(y))`.
- Partition-dim broadcasting is stricter on tensor-tensor arith — matching
  partition dims required.

See the [suite-wide guide](https://github.com/trnsci/trnsci/blob/main/docs/developing_kernels.md)
for rationale and cross-library migration notes.
