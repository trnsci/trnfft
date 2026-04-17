# Kahan on-silicon characterization

**Issue:** [#58](https://github.com/trnsci/trnfft/issues/58)
**Hardware:** trn1.2xlarge, Neuron SDK 2.29.0
**Test:** `tests/test_precision_modes.py::TestKahanCharacterization`

## Background

`trnfft.set_precision("kahan")` switches the NKI butterfly to a Dekker 2Prod
compensated variant (`butterfly_stage_kernel_kahan`, ~2× butterfly op count)
and applies Dekker compensation to the chirp multiplies in the Bluestein chain.

On CPU, `"kahan"` equals `"fast"` — the Dekker compensation in the butterfly
only engages in the NKI kernel, and chirp-multiply compensation doesn't help on
a host with FP64.  The on-silicon question: does the kahan butterfly actually
reduce Bluestein FP32 error on Trainium, and at what timing cost?

## Method

Run `trnfft.fft(x)` for `N ∈ {997, 1009, 8193}` (non-power-of-2 → Bluestein)
in both `"fast"` and `"kahan"` modes.  Compare each against a `scipy.fft.fft`
fp64 reference.  Hardware: trn1.2xlarge, NKI path (`set_backend("nki")`).

## Results

*Placeholder — fill with `pytest --capture=no -v tests/test_precision_modes.py`
output after hardware run.*

| N    | fast error (rel) | kahan error (rel) | kahan improvement |
| ---- | ---------------- | ----------------- | ----------------- |
| 997  |                  |                   |                   |
| 1009 |                  |                   |                   |
| 8193 |                  |                   |                   |

## Timing overhead

*Placeholder — add wall-time comparison after hardware run.*

| N    | fast (ms) | kahan (ms) | overhead |
| ---- | --------- | ---------- | -------- |
| 997  |           |            |          |

## Analysis

*Fill after results.*

Expected behavior: the Dekker 2Prod compensation reduces per-butterfly-stage
accumulation error, which should propagate to a smaller final Bluestein error vs
the fp64 reference.  Whether the reduction is meaningful in practice (e.g. from
~1e-2 to ~1e-3) or marginal is the empirical question this characterization
answers.

## Recommendation

*Fill after results.*

Candidate recommendation: use `"kahan"` for Bluestein chains (non-power-of-2 N)
when FP32 error at N > 500 matters and the ~2× butterfly timing overhead is
acceptable.  For power-of-2 N and batch workloads where DFT-GEMM or butterfly
with high batch utilization is the bottleneck, `"fast"` is appropriate.
