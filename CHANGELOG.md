# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.0] - 2026-04-30

### Added

- **Stage-parallel FFT** (`_stage_parallel_fft` in `trnfft/nki/multicore.py`). Enables
  single-transform multi-core execution for composite N via the row-column (Cooley-Tukey
  2D) decomposition. No NKI inter-core communication primitive required — the twiddle
  multiply between phases is the "inter-core exchange" and runs in FP32 on the host.

  Algorithm for N = n1 × n2:
  1. Reshape x to (n1, n2)
  2. Column DFTs: batch of n2 size-n1 FFTs (dispatched via `_batch_split_fft`)
  3. Twiddle: Z[k1, l] × exp(sign·2πi·l·k1/N)
  4. Row DFTs: batch of n1 size-n2 FFTs
  5. Column-major flatten: X_flat[k] = X_2d[k%n1, k//n1]

  Only beneficial for N > ~2^17 where single-core SBUF becomes the bottleneck.
  For prime N, `multi_core_fft` raises `NotImplementedError` with a clear message.

- `_factorize(n)` — returns (n1, n2) with n1*n2=n and n1 ≈ sqrt(n); raises ValueError
  for prime n.
- `scripts/run_precision_characterization.sh` — focused SSM script for running
  `TestOzakiHQCharacterization` on trn1 (5–10 min vs 30 min for full neuron suite).
- Extended `tests/test_multicore.py` with `TestFactorize`, `TestStageParallelFFT`, and
  `TestMultiCoreSingleTransform` classes.

### Changed

- `multi_core_fft`: single-transform inputs now route to `_stage_parallel_fft` for
  composite N instead of unconditionally raising `NotImplementedError`.

## [0.20.0] - 2026-04-29

### Added

- **Multi-NeuronCore batch-split FFT** (`trnfft/nki/multicore.py`). Replaces the
  v0.19 scaffold passthrough with a genuine batch-splitting implementation.
  `set_multicore(True, num_cores=N)` routes batched inputs through `_batch_split_fft`,
  which splits the batch dimension across N NeuronCores (or CPU threads in test mode)
  and reassembles the result.

  On Neuron hardware with `torch_neuronx` available: compiles a thin `_FFTModule`
  wrapper with `torch_neuronx.trace`, then dispatches via `torch_neuronx.DataParallel`.
  Compiled models are cached per `(n, inverse, num_cores)` key so the first call pays
  the compilation cost and subsequent calls do not.

  On CPU (no `torch_neuronx`): processes shards sequentially — architecturally correct,
  no hardware parallelism. All CPU tests pass.

  Single-transform inputs still raise `NotImplementedError`; stage parallelism requires
  inter-core all-reduce infrastructure and is deferred.

- `_resolve_num_cores(batch_size)` — resolves actual core count from requested count,
  `torch_neuronx.get_neuron_device_count()`, or CPU fallback (min(2, batch)).
- `tests/test_multicore.py` — CPU correctness tests covering shape contracts, roundtrip,
  core clamping, and API surface.

## [0.19.0] - 2026-04-27

### Added

- **2-level Ozaki-scheme DFT-GEMM** (`precision="ozaki_hq"`). Extends v0.18's 1-level
  scheme by using a 3-way x split (keeping FP32 residuals between levels) plus a 2-way
  W split — 6 BF16 matmuls total. The key constraint from v0.18: once a value is
  quantised to BF16, its residual has zero mantissa bits for a second split; the FP32
  intermediate is what enables the second level.

  Expected accuracy: O(sqrt(N)·u_bf16^4) ≈ 2e-9 rel error at N=64 — ~1000× better
  than 1-level Ozaki (~1.6e-5) and near-FP64 without CPU roundtrip.
  Hardware results (trn1, SDK 2.29, 2026-04-27):

  | N   | oz_hq (µs) | ozaki (µs) | BF16 (µs) | hq/ozaki | hq/fp32 |
  | --- | ---------- | ---------- | --------- | -------- | ------- |
  | 64  | 6 252      | 3 225      | 1 178     | 1.94×    | 3.41×   |
  | 128 | 6 313      | 3 316      | 1 214     | 1.90×    | —       |
  | 256 | 6 417      | 3 451      | 1 300     | 1.86×    | 3.41×   |

  The ~1.9× overhead vs 1-level Ozaki (not 2×) reflects kernel-call pipelining.
  Hardware precision (trn1, SDK 2.29.0, 2026-04-30): both `ozaki` and `ozaki_hq`
  measure ~1.7e-3 rel error at N=64 — equivalent to a single-pass BF16 DFT-GEMM.
  The ORO split corrects input quantization error (BF16 W and x), but the Trainium
  Tensor Engine rounds BF16 products before PSUM accumulation; that per-product
  rounding dominates and is not captured by the split. The scheme works as theorised
  on CPU (where the complex_gemm_bf16 fallback promotes inputs to FP32 before matmul),
  but not on hardware. Throughput improvement from the multi-term structure remains
  valid (3× / 6× BF16 latency); only the precision claim is revised.

  | mode      | trn1 rel error (N=64) | trn2 rel error (N=64) | CPU rel error (N=64) |
  | --------- | --------------------- | --------------------- | -------------------- |
  | bf16      | ~1.5e-3               | ~1.5e-3               | ~2.2e-3              |
  | ozaki     | ~1.7e-3               | ~1.7e-3               | ~1.6e-5              |
  | ozaki_hq  | ~1.7e-3               | ~1.7e-3               | ~1.4e-7              |

  trn2.3xlarge (sa-east-1, SDK 2.29.0, 2026-05-01): identical to trn1. Both
  hardware generations round BF16×BF16 products to BF16 before PSUM. The NKI
  Bootcamp phrase "BF16 products computed at full precision internally" refers to
  the FP32 PSUM accumulator, not the individual product precision. trn3 MXFP8
  (`nisa.nc_matmul_mx`) with FP32 accumulation is the next test point.

- `_ozaki_split_3way_bf16(x)` — 3-way ORO split returning (x_h1, x_h2, x_h3) BF16,
  using FP32 for the intermediate residual before the second quantisation.
- `_FORCE_OZAKI_HQ = False` bench toggle; `TestFFT1DOzakiHQ` benchmark class.

### Changed

- `trnfft/nki/dispatch.py`: removed unused `_ozaki_split_bf16` and `complex_gemm_ozaki`
  (consolidated into `trnfft/fft_core.py` alongside the new 2-level helpers).

## [0.18.0] - 2026-04-23

### Added

- **Ozaki-scheme DFT-GEMM** (`precision="ozaki"`). 1-level BF16 split of W and x
  into high/low BF16 parts (Ogita–Rump–Oishi split), followed by 3 BF16 matmuls
  (`W_h@x_h + W_h@x_l + W_l@x_h`) accumulated in FP64. Output is FP32.

  Expected accuracy: O(sqrt(N) × u_bf16²) relative error ≈ 8–16e-6 at N=64–256.
  Cost: 3 × BF16 DFT-GEMM ≈ 2× FP32 DFT-GEMM. Faster than `"double"` (no CPU
  roundtrip). Hardware validation pending.

  The deterministic error bound (vs IR-1's feedback loop) makes this suitable for
  the eventual `target_forward_error` API contract.

  **What doesn't work yet:** A 2-level split (for O(u_bf16^4) accuracy) requires
  keeping the FP32 residual through multiple stages before the final BF16
  quantisation. The residual of a BF16 value is exactly representable in FP32 but
  has zero BF16 mantissa bits remaining — you can't split it again in BF16. Deferred
  to v0.19.

- `_ozaki_split_bf16(x)` helper in `trnfft/nki/dispatch.py` and `trnfft/fft_core.py`.
- `complex_gemm_ozaki(a, b)` in `trnfft/nki/dispatch.py`.
- `_FORCE_OZAKI = False` bench toggle; `TestFFT1DOzaki` benchmark class.

  Hardware results (trn1, SDK 2.29, 2026-04-25):

  | N   | Ozaki (µs) | BF16 (µs) | FP32 DFT-GEMM | oz/bf16 | oz/fp32 |
  | --- | ---------- | --------- | ------------- | ------- | ------- |
  | 64  | 3 241      | 1 179     | ~1 833        | 2.75×   | 1.77×   |
  | 128 | 3 291      | 1 216     | —             | 2.71×   | —       |
  | 256 | 3 466      | 1 302     | ~1 882        | 2.66×   | 1.84×   |

  Cost: ~2.7× single-pass BF16; ~1.8× FP32 DFT-GEMM. On-chip (no CPU roundtrip).

  **Debugging note (v0.18):** Eight hardware benchmark attempts produced no timing data
  before the root cause was identified: the `_FORCE_OZAKI` bench toggle called
  `_fft_via_ozaki(x, inverse, levels=2)` even after the `levels` parameter was removed
  from the function signature. Every attempt threw `TypeError` before timing started.
  The data-dependency trick (`0 * hh.mean()`) added in earlier attempts is kept because
  it correctly forces sequential XLA graph execution for the 3 cross-term kernels.

## [0.17.0] - 2026-04-22

### Added

- **BF16 DFT-GEMM with PSUM-FP32 output** (`precision="bf16"`). A new NKI kernel
  `_complex_gemm_kernel_bf16` takes BF16 inputs, accumulates to FP32 PSUM (hardware
  invariant), and stores FP32 output — skipping the `nl.cast` back to BF16 that the
  existing kernel performs. This is the "PSUM is a free FP32 accumulator" architectural
  principle applied to DFT-GEMM. Hardware results (trn1, SDK 2.29, 2026-04-22):

  | N   | BF16 DFT-GEMM (µs) | FP32 DFT-GEMM (µs) | Speedup |
  | --- | ------------------- | ------------------- | ------- |
  | 64  | 1 189               | ~1 833 (v0.12)      | ~1.5×   |
  | 128 | 1 208               | —                   | —       |
  | 256 | 1 310               | ~1 882 (v0.12)      | ~1.4×   |

  BF16 W quantisation limits accuracy to ≈1e-3 rel error at N=256. For near-FP32
  accuracy, use `precision="bf16_refined"` (one correction step).
  Dispatched for N ≤ 256 when `precision="bf16"`.

- **Iterative FFT refinement** (`precision="bf16_refined"`). One correction step on top
  of the BF16 path drives accuracy to near-FP32:

  .. code-block:: text

      X̂ = fft_bf16(x)          # BF16 compute, FP32 PSUM output
      r = x − IFFT(X̂)          # FP32 residual
      X̂ = X̂ + fft_bf16(r)     # correction

  This is the first implementation of BF16 FFT with PSUM-FP32 residual correction on a
  production deterministic systolic array. Cost: 2 BF16 FFTs + 1 IFFT.
  Dispatched for N ≤ 256 when `precision="bf16_refined"`. Hardware validation pending.

- `_FORCE_BF16_GEMM = False` bench toggle; `TestFFT1DBF16` benchmark class.

## [0.16.0] - 2026-04-21

### Added

- **Mixed-radix Stockham FFT** (`_fft_via_stockham_nki_mixed`). Computes the optimal
  `[8^a, 4^b]` stage decomposition for any power-of-2 N. Hardware results (trn1, SDK 2.29,
  2026-04-21):

  | N    | Mixed (µs) | Previous path    | Previous (µs) | Delta |
  | ---- | ---------- | ---------------- | ------------- | ----- |
  | 1024 | 5 863      | radix-4 (5 stages) | 6 884       | −15%  |
  | 2048 | 5 984      | butterfly (11 stages) | ~8 600 (est.) | ~−30% |

  No new NKI kernels: driver interleaves the existing `stockham_radix8_w8_kernel` (for r=8
  stages, Tensor engine W₈) and `stockham_radix4_stage_kernel` (for r=4 stages). Auto-
  dispatched for power-of-2 N where mixed gives fewer stages than the current best path.
  `stockham_mixed_radix` CPU reference added to `trnfft/stockham.py`.

- **`hfft` and `ihfft`** — completes the `torch.fft` API surface. These were the only two
  functions missing:
  - `hfft(x, n)`: FFT of a Hermitian-symmetric signal; returns real output of length `n`.
    Implemented as `irfft(x.conj(), n) * n` (irfft uses backward norm; hfft uses forward).
  - `ihfft(x, n)`: inverse of `hfft`; returns one-sided complex spectrum of length `n//2+1`.
    Implemented as `conj(rfft(x, n)) / n`.

- **`TestKahanButterflyCharacterization`** in `tests/test_precision_modes.py` — hardware
  test that measures actual on-silicon rel error for `precision="kahan"` butterfly at
  N ∈ {256, 512, 1024, 4096}. Forces butterfly path (`_DFT_GEMM_THRESHOLD=0`) and prints
  fast vs kahan error + improvement ratio. Fills the known gap in `precision.py` docstring
  (previously said "Target ~1e-3" without measured data).

## [0.15.0] - 2026-04-20

### Added

- **Thread B: radix-8 Stockham FFT with Tensor-engine W_8** (`_fft_via_stockham_nki_r8`).
  New coverage for N=512 (= 8³, previously routed to 9 butterfly stages → now 3 radix-8
  stages). Improved coverage for N=4096 (= 8⁴, previously 6 radix-4 stages → now 4
  radix-8 stages). Auto-dispatched for all power-of-8 N > 256.

  Per-stage structure: twiddle multiply via PyTorch element-wise on XLA device, then
  `stockham_radix8_w8_kernel` applies the W_8 DFT matrix via `nc_matmul` (Tensor engine).
  W_8 entries are non-trivial (±√2/2 ± i√2/2), so the Tensor engine earns its keep
  unlike W_4 (which has {1, -1, i, -i} coefficients and only needs adds/swaps).

  Hardware results (trn1, SDK 2.29, 2026-04-20):

  | N    | Radix-8 (µs) | Radix-4 (µs) | Butterfly (µs) | vs r4  | vs butterfly |
  | ---- | ------------ | ------------ | -------------- | ------ | ------------ |
  | 64   | 3 402        | 4 254        | 4 767          | −20%   | −29%         |
  | 512  | 4 483        | —            | ~6 600 (est.)  | —      | ~−32%        |
  | 4096 | 5 917        | 8 424        | 9 387          | −30%   | −37%         |

- `_is_power_of_eight(n)`, `_FORCE_STOCKHAM_R8` bench toggle, `stockham_radix8` CPU
  reference, `_w8_matvec` helper (`trnfft/stockham.py`).
- `stockham_radix8_w8_kernel` NKI kernel (`trnfft/nki/stockham.py`).

### Engineering note — NKI scratch buffer constraint

Initial implementation used a kernel-internal `nl.ndarray(buffer=nl.shared_hbm)` scratch
to bridge twiddle-multiply and W_8 matmul phases in one kernel call. This failed NEFF
compilation: `nl.load_transpose2d` in NKI 0.3.0 only accepts **function-argument** HBM
tensors as source, not kernel-local allocations. Fixed by moving twiddle multiply to the
PyTorch driver (XLA element-wise op) and making the NKI kernel W_8-only.

## [0.14.0] - 2026-04-20

### Fixed

- **`precision="double"` was silently ignored for power-of-2 FFTs.** `_fft_via_gemm`
  received no precision parameter; users calling `set_precision("double")` for small-N
  accuracy got FP32 DFT-GEMM (~1e-3 error) instead of FP64 (~1e-6 for FP32 inputs,
  ~1e-12 for FP64 inputs). Fixed by routing `precision="double"` through a new
  `_fft_via_gemm_double()` function that:
  - Computes `W @ x` on CPU in FP64 (bypasses NKI — Trainium PSUM is always FP32)
  - Activates for N ≤ `_DOUBLE_GEMM_THRESHOLD = 1024`
  - Casts result back to input dtype (consistent with Bluestein "double" behaviour)
  - For N > 1024 in "double" mode: falls through to NKI Stockham (~1e-4 FP32)

### Added

- `_stockham_perm_indices(log4n, B_pad, n)` utility in `fft_core.py` — precomputes
  flat pack/unpack index tensors for all Stockham stages (CPU int64). Not used in the
  hot path (see note below) but available for testing, profiling, and future
  NKI gather experiments when cross-partition addressing becomes available.
- `trnfft/nki/stockham.py`: documents `stockham_radix4_fused_kernel` as a named stub
  (Thread C phase 2) with the architectural constraint that blocks it: cross-partition
  scatter in NKI is not supported for the N values where Stockham dispatches.

### Research note — Thread C phase 1 (gather approach measured slower)

Hardware bench (trn1, SDK 2.29, 2026-04-18) showed that replacing `permute+contiguous`
(transpose HLO) with a precomputed flat-index gather (GatherOp HLO) is **11–39% slower**
across all tested N:

| N    | v0.13 (µs) | v0.14 gather (µs) | delta |
|------|------------|-------------------|-------|
| 16   | 3 121      | 3 493             | +12%  |
| 64   | 4 322      | 4 796             | +11%  |
| 256  | 5 700      | 6 103             | +7%   |
| 1024 | 6 850      | 7 950             | +16%  |
| 4096 | 8 632      | 11 991            | +39%  |

Neuron's transpose HLO is hardware-optimized; the GatherOp for non-affine indices is not.
The driver reverts to the `permute+contiguous` approach. `_stockham_perm_indices` is kept
as a utility. The permute overhead (~97 µs/stage) is an irreducible cost until true
SBUF-resident stage fusion is possible (Thread C phase 2).

## [0.13.0] - 2026-04-15

### Added

- **NKI 0.3.0 (Neuron SDK 2.29) migration + CPU simulator dispatch** (#59). `trnfft` now targets the stable `nki` package namespace (`import nki` instead of `import neuronxcc.nki`) and adopts the 0.3.0 calling convention (`nisa.nc_matmul` kwargs-only; `nisa.tensor_copy` for PSUM→SBUF). Kernels run in a new simulator mode via `TRNFFT_USE_SIMULATOR=1`, routing through `nki.simulate(kernel)(numpy_args)` on CPU. Catches Python-trace-level errors (bad kwargs, dropped ops, shape mismatches) without round-tripping to Trainium. MLIR verifier errors remain hardware-only. Hardware-validated on trn1.2xlarge, SDK 2.29.0, 2026-04-15.
- New `nki-simulator` job in `.github/workflows/ci.yml` running simulator-marked tests on `ubuntu-latest` — first correctness gate for kernel changes that doesn't need AWS access.
- New `tests/test_nki_sim.py` with simulator-backed tests for `_complex_gemm_kernel`, `_complex_mul_kernel`, and the butterfly FFT path.
- New `scripts/run_simulator_tests.sh` mirroring `run_neuron_tests.sh` but with `TRNFFT_USE_SIMULATOR=1`.
- `docs/developing_kernels.md` — trnfft-specific kernel dev pattern.

- **Stockham radix-4 POC** (Thread B). CPU reference (`trnfft/stockham.py`) and NKI kernel (`trnfft/nki/stockham.py`) validated on hardware. Precision-safe to N=4096+ (log₄(N) FP32 accumulation; CPU-reference error ~5e-5 at N=4096). Available via `_FORCE_STOCKHAM=True` bench toggle — not dispatched by default (see "Known limitations" below).

- **Measured v0.12 DFT-GEMM wins on batched FFT + STFT.** Head-to-head bench on trn1 shows the architectural thesis pays off end-to-end:
  - Batched FFT `(B=32, N=128)`: **15.8× faster** than the large-N butterfly path; 6.3× faster than the PyTorch fallback.
  - Batched FFT `(B=32, N=256)`: **14.3× faster** / 11.3× vs PyTorch.
  - STFT `n_fft=128`: **13.1× faster** than `n_fft=512` butterfly; 6.2× faster than PyTorch.
  - STFT `n_fft=256`: **12.5× faster** / 10.5× vs PyTorch.

### Changed

- `pyproject.toml` `[neuron]` extra pins `nki>=0.3.0` (was `neuronxcc>=2.24`). Users on pre-2.29 SDKs must upgrade the DLAMI (`terraform apply` picks up the new AMI automatically).
- Existing `test` matrix runs with `-m "not neuron and not nki_simulator"` so each test runs on exactly one CI job.
- **SDK 2.29 improved butterfly performance by 1.3–2.1× vs SDK 2.24.** E.g. N=256 butterfly: 9 862 μs → 6 067 μs; N=1024: 15 746 μs → 7 399 μs. DFT-GEMM speedup over butterfly at N=256 narrowed from 5.2× to 3.0×; DFT-GEMM is still the dominant path at N ≤ 256.

### Changed (continued)

- **Stockham twiddle precomputation (SHA `a74b697`) makes Stockham the default path for power-of-four N > 256.** Precomputing all log₄(N) twiddle tensors before the stage loop eliminates per-stage H→D transfer overhead — the dominant cost term in the pre-precompute baseline. Hardware bench (trn1, SDK 2.29, 2026-04-17): Stockham is **6–9% faster than butterfly** at all tested N (16–4096). `trnfft` now auto-dispatches power-of-four N above `_DFT_GEMM_THRESHOLD` through the Stockham kernel. `_DFT_GEMM_THRESHOLD = 256` unchanged — DFT-GEMM is still ~3× faster than Stockham at N ≤ 256.

| N    | Stockham (μs) | Butterfly (μs) | Delta    |
| ---- | ------------- | -------------- | -------- |
| 16   | 3 121         | 3 337          | −6%      |
| 64   | 4 322         | 4 767          | −9%      |
| 256  | 5 700         | 6 067          | −6%      |
| 1024 | 6 850         | 7 399          | −7%      |
| 4096 | 8 632         | 9 387          | −8%      |

## [0.12.0] - 2026-04-13

### Added

- **DFT-as-GEMM fast path for small-to-medium FFT on Trainium.** `_fft_via_gemm` routes FFT onto the Tensor engine via the existing `_complex_gemm_kernel` — one matmul instead of log₂(N) butterfly-stage launches. Dispatches automatically for N ≤ 256 when running on NKI (threshold precision-bound, not perf-bound — see below). CPU path unaffected.
- New design note `docs/design-notes/fft-is-a-gemm.md` documenting the architectural thesis: FFT on Trainium is a problem about kernel launches and Tensor-engine utilization, not arithmetic complexity. Includes head-to-head sweep at N ∈ {8..2048}.
- 9 new CPU tests in `TestDFTGEMM` covering matches-numpy, roundtrip, and batched cases.

### Changed

- At small N, `trnfft.fft(x)` on Trainium is dramatically faster. Measured on trn1 (NKI 2.24.5133.0) via head-to-head bench (forcing each kernel path explicitly):

| N    | DFT-GEMM (μs) | Butterfly (μs) | Speedup |
| ---- | ------------- | -------------- | ------- |
| 8    | 1716          | 3832           | 2.2×    |
| 64   | 1833          | 6997           | 3.8×    |
| 256  | 1882          | 9862           | 5.2×    |
| 1024 | 2954          | 15746          | 5.3×    |
| 2048 | 17500         | 23819          | 1.4×    |

DFT-GEMM time is roughly constant ~1.7-2.1 ms through N=512 (launch overhead dominates), while butterfly grows cleanly linear in log₂(N) at ~1 ms per stage. Hard perf cliff at N=2048 where partition-dim underutilization and K-tile count together eat the launch-count win.

### Known limitations

- The 256 threshold is **precision-bound**: FP32 `nc_matmul` at N=1024 accumulates ~2.2% relative error, which exceeds the test suite's 1e-3 tolerance. The performance crossover is N ≈ 1024–2048, so there's untapped win at N=512–1024 that's currently unreachable in FP32. Stockham radix-r (v0.13 Thread B) is the structurally-correct path to extend past 256 without paying O(N²) matmul accumulation.
- Single-FFT case (M=1 batch) at large N wastes 127 of 128 partitions. Batched FFT and STFT get better partition utilization automatically because they flatten to `(B, N)` before dispatch.

## [0.11.0] - 2026-04-13

### Added

- **Precision modes for Bluestein FFT** (#52, partial). New `trnfft.set_precision(mode)` API selects between:
  - `"fast"` (default, unchanged): existing FP32 path.
  - `"double"`: promotes Bluestein host math to FP64, casts back on exit. Empirically 5.5e-13 rel error at n=500 and 7.5e-9 at n=8193 — 10+ orders of magnitude better than `"fast"` (which accumulates ~2.5e-4 rel at n=500, ~2.2e-3 at n=8193). Power-of-2 Cooley-Tukey is unaffected; FFTs inside Bluestein stay on PyTorch because NKI kernels are FP32-only.
  - `"kahan"`: Dekker 2Prod compensated complex multiply at the two chirp multiplies and at the inner Y*H product; NKI butterfly kernel has a matching `butterfly_stage_kernel_kahan` variant (~2× butterfly op count). On CPU `"kahan"` equals `"fast"` because the chirp multiplies aren't the dominant error source — the 3-FFT butterfly chain is. The kahan kernel compiles and matches fast-mode output on silicon (validated on trn1 against NKI 2.24.5133.0); on-silicon precision characterization is deferred to v0.12.

- 10 new CPU tests in `TestBluesteinsPrecision` covering fast/double correctness and the precision-setter API; 1 new neuron-marked test `test_kahan_butterfly_compiles_and_matches_fast` exercising the Dekker butterfly kernel on trn1.

- Architecture doc gains a Precision modes section with per-mode error/cost table.

### Changed

- Precision is threaded through `fft_core` → `_cooley_tukey` → `_cooley_tukey_nki` → `_cooley_tukey_nki_nograd`, and through the `_FFTFn` autograd wrapper so gradients respect the chosen mode.

## [0.10.1] - 2026-04-13

### Fixed

- **NKI kernels now preserve autograd** (#56). Prior to this release, any training loop that touched a NKI code path (`ComplexLinear`, `trnfft.fft`/`stft` with `set_backend("nki")`, `complex_gemm`, or `complex_mask_apply`) silently detached the autograd graph because `@nki.jit`-decorated kernels returned raw tensors from `nl.shared_hbm` with no `grad_fn`. `loss.backward()` raised `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`. The workaround was `trnfft.set_backend("pytorch")`, which bypassed NKI entirely.

  Fix: new `trnfft/nki/autograd.py` with `torch.autograd.Function` subclasses — `_ComplexMulFn`, `_ComplexGEMMFn`, `_ComplexLinearFn`, `_FFTFn` — whose `forward` calls the NKI kernel and whose `backward` emits the analytic adjoint. For complex mul / GEMM / Linear the adjoint is the conjugate-transposed version of forward; for FFT the adjoint is IFFT scaled by `n` (and vice versa). Backward is implemented via standard PyTorch ops on the same device as forward; backward-on-NKI is a possible future optimization.

- Three neuron-marked gradient tests in `TestNKIGradients` cover ComplexLinear, 1D FFT, and STFT end-to-end training paths on trn1.2xlarge.

## [0.10.0] - 2026-04-13

### Added

- `rfft2(input, s=None)` — 2D FFT of a real signal; output shape `(..., M, N//2 + 1)` (Closes #48).
- `irfft2(input, s=None)` — inverse of `rfft2`, Hermitian-reconstructs the last dim.
- `rfftn(input, s=None, dim=None)` — N-D FFT of a real signal.
- `irfftn(input, s=None, dim=None)` — inverse of `rfftn`.
- 6 new CPU tests in `TestRFFTnD` covering vs-numpy correctness, roundtrips, default shape inference, and `s=` padding/truncation.

### Changed

- API coverage note in README and docs/index.md updated: trnfft now implements **13 of ~15** common `torch.fft` transforms. Only `hfft` and `ihfft` (Hermitian-input variants) remain unimplemented.

## [0.9.0] - 2026-04-13

### Fixed

- `_complex_mul_kernel` now compiles for shapes that tile multiple times in the free dim (e.g., 1024×512 = 4096-free → 8 tiles). The prior `f_end = min(f_off + FMAX, free)` pattern failed inside `nl.affine_range` because NKI 2.24 can't evaluate `min()` symbolically. Replaced with a constant chunk size plus an explicit divisibility assert — same pattern as the v0.8.0 butterfly fix. All 70 benchmark cases now pass on trn1.2xlarge (#39).

### Changed

- **Documentation refresh** for the matured `trn-*` suite:
  - README status banner updated from v0.1.0 "scaffolded" text to v0.8.0 summary citing benchmark wins.
  - Related Projects table expanded to all 6 siblings + umbrella meta-package, with PyPI release versions.
  - CLAUDE.md naming convention section removes `(planned)` markers; adds trnsparse, trntensor, trnsci.
  - docs/index.md clarifies API coverage (9 of 12 `torch.fft` functions; `hfft`, `ihfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn` tracked as roadmap).

### Roadmap (moved to v0.10.0 milestone)

- SBUF-resident dispatch for small-op overhead (#40) + investigation spike (#47)
- `torch.fft` API parity decision (#48)
- NKI ComplexConv1d kernel (#49)
- BF16 / FP16 support across kernels (#50)

## [0.8.0] - 2026-04-12

### Added

- Batched NKI butterfly kernel: `butterfly_stage_kernel` now accepts `(B, n)` input and vectorizes across the batch dim within a single kernel invocation per stage. Removes the per-row Python loop in `_cooley_tukey_nki` that was the dominant cost on multi-call paths (fftn, fft2, batched 1D FFT, STFT).
- `docs/benchmark_results/v0.8.0.txt` — full v0.8.0 hardware results from trn1.2xlarge.

### Changed

- `_cooley_tukey_nki` pads the batch dim up to the next multiple of PMAX (128) only when `B` is not already a power of 2. Unbatched FFT (`B=1`) and power-of-2 batched FFT take the zero-copy path. Non-power-of-2 cases (STFT's `num_frames=33`) pad with zeros, run, then discard the padding.
- Repository rehomed from `scttfrdmn/trnfft` to `trnsci/trnfft`. URLs updated throughout (`README.md`, `pyproject.toml`, `mkdocs.yml`, `docs/installation.md`, `infra/terraform/main.tf`, CHANGELOG footer links). GitHub transparently redirects the old URLs.
- Docs Pages URL migrated from `scttfrdmn.github.io/trnfft/` to `trnsci.github.io/trnfft/`.
- Author email in `pyproject.toml` switched to GitHub noreply.

### Fixed

- `stft` frame extraction no longer uses `torch.Tensor.unfold`, which has no XLA backend implementation in torch-xla 2.9 / torch-neuronx. Replaced with explicit `torch.arange`-based index construction, device-agnostic across CPU, CUDA, MPS, and XLA/Trainium (PR #44).
- `_complex_mul_kernel` wrapper now forces input tensors contiguous before reshape to avoid NKI 2.24 compile errors on non-contiguous views.

### Performance (v0.7.0 → v0.8.0 on trn1.2xlarge; NKI path)

| Operation | v0.7.0 | v0.8.0 | Speedup |
|---|---:|---:|---:|
| fftn 32×64×64 | 52.3 s | 70.8 ms | 738× faster |
| fft2 256×256 | 5.0 s | 45 ms | 111× faster |
| fft2 1024×1024 | 32.4 s | 545 ms | 59× faster |
| batched FFT (128×1024) | 2.07 s | 52 ms | 39× faster |
| batched FFT (32×1024) | 520 ms | 25 ms | 21× faster |
| STFT (16k samples) | 765 ms | 28 ms | 27× faster |

**trnfft-NKI now beats vanilla `torch.fft.fft` for batched FFT and STFT.** Single 1D FFT and Bluestein paths are unchanged from v0.7.0 (already batched across groups). Full table and caveats in `docs/benchmarks.md`.

### Known issues

- `#39` Complex mask kernel still fails on 1024×512 shape — contiguity fix was insufficient; actual root cause still TBD.
- `#40` SBUF-resident dispatch for small ops — deferred to v0.9.0.

## [0.7.0] - 2026-04-12

### Added

- Three-baseline benchmark suite (`benchmarks/bench_fft.py`) — NKI vs trnfft-PyTorch vs raw `torch.*` — covering 1D FFT, fft2, fftn, batched FFT, Bluestein, STFT, complex GEMM, ComplexLinear, and complex mask apply across multiple shapes (70 cases total).
- `scripts/run_benchmarks.sh` — starts the trn1 instance, runs the suite under SSM, fetches results, stops the instance.
- `scripts/bench_to_md.py` and `scripts/bench_text_to_md.py` — convert pytest-benchmark output (JSON or pretty-printed text) into the markdown table for `docs/benchmarks.md`.
- `docs/benchmarks.md` — methodology, results table, and findings section. Linked from README and mkdocs nav.
- `docs/benchmark_results/v0.7.0.txt` — raw captured output from the v0.7.0 hardware run for reproducibility.

### Findings (v0.7.0 hardware run on trn1.2xlarge)

- NKI is 2-11× faster than the trnfft PyTorch fallback for single 1D FFT (n ≥ 256), 2-10× for Bluestein, and 3-4× for complex GEMM ≥ 1024².
- Multi-call paths (fftn, fft2, batched 1D FFT, STFT) are catastrophically slow on NKI right now (50-1600× slower) because `_cooley_tukey_nki` loops over batch rows in Python — filed as #38 (batched butterfly kernel).
- Small ops are dispatch-bound; host CPU wins below per-op thresholds.
- Detailed table and analysis in `docs/benchmarks.md`.

## [0.6.0] - 2026-04-12

### Added

- Fused NKI 4-real-matmul kernel `_complex_linear_kernel` for `ComplexLinear` — loads activations once, streams W_real/W_imag against both phases. `ComplexLinear.forward()` now dispatches to NKI when `set_backend("nki")` is active (previously always took the 4 separate `nn.Linear` PyTorch path).
- `complex_linear()` exposed in `trnfft.nki` package.
- 9 new neuron-marked tests, bringing hardware coverage from 4 to **13 tests**: ComplexLinear NKI vs PyTorch, ComplexConv1d/ModReLU PyTorch fallback, STFT shape + ISTFT roundtrip, fft2 (2D), fftn (3D), batched FFT, Bluestein arbitrary sizes (7, 13, 100, 127), FFT at 4096/16384 to exercise multi-partition tiling.
- GEMM hardware shape coverage extended to (512, 512) and (1024, 1024).

### Changed

- `_cooley_tukey_nki` now handles batched input (N-D) by flattening leading dims and iterating row-by-row, so STFT, fft2, fftn, and batched 1D FFT all route through the NKI butterfly kernel.
- Replaced deprecated `xm.xla_device()` with `torch_xla.device()` (eliminates 12 deprecation warnings per neuron test run).

## [0.5.0] - 2026-04-12

### Added

- Terraform module (`infra/terraform/`) for provisioning a Trainium CI instance with SSM access.
- `docs/aws_setup.md` and `scripts/run_neuron_tests.sh` for running neuron-marked tests locally via `AWS_PROFILE=aws`.
- Hardware compatibility matrix in `docs/installation.md` documenting validated Neuron SDK and AMI versions.

### Changed

- All three NKI kernels validated and passing on real Trainium hardware (trn1.2xlarge, neuronxcc 2.24.5133.0).
- NKI butterfly kernel rewritten with 2D `(num_groups, m)` partition-dim tile layout. Twiddle factors are pre-broadcast on the host to `(num_groups, half)` so element-wise ops have matching partition dims. Outputs are allocated and returned by the kernel (NKI 2.24 parameters are immutable). Constant-size partition tiles (no `min()` in `affine_range`).
- `_complex_gemm_kernel` updated to use the NKI 2.24 calling convention (`psum[...] += nisa.nc_matmul(stationary, moving)` returning a PSUM tile) and `nl.load_transpose2d` for the stationary A tile. Uses `nl.negative` + `+=` instead of `-=` (unsupported inside `affine_range`).
- `_complex_mul_kernel` reshapes inputs to `(128, free)` to satisfy the partition-dim ≤ 128 constraint; falls back to PyTorch for inputs whose total size isn't divisible by 128.
- All NKI dispatch wrappers move tensors to/from the XLA device (`xm.xla_device()`) since NKI kernels require XLA tensors.
- Pinned `neuronxcc>=2.24` and `torch-neuronx>=2.9` in the `neuron` extra. Kernels do not compile against older Neuron SDKs.
- `neuron.yml` GitHub Actions workflow removed; AWS access is now local-only via `scripts/run_neuron_tests.sh` with `AWS_PROFILE=aws`.

## [0.4.0] - 2026-04-11

### Added

- MkDocs documentation site with pages for installation, quickstart, API reference, and architecture.
- GitHub Pages deployment workflow.
- PyPI publishing workflow (OIDC trusted publishers, sdist + wheel on release).
- Benchmark suite (`benchmarks/bench_fft.py`) for 1D/2D FFT, STFT, and complex GEMM.
- Multi-NeuronCore parallelism scaffold (`trnfft.nki.multicore`) with data-parallel batch split.
- PyPI, Python version, license, and docs badges in README.

### Changed

- LICENSE replaced with full Apache 2.0 text (Copyright 2026 Scott Friedman).
- `pyproject.toml` dev deps now include mkdocs and mkdocs-material.

## [0.3.0] - 2026-04-11

### Added

- NKI complex GEMM kernel with stationary tile reuse (ported from neuron-complex-ops).
- NKI fused element-wise complex multiply kernel.
- NKI butterfly kernel wired into Cooley-Tukey FFT path.
- `nki_backend` test fixture and 4 neuron-marked tests for on-hardware validation.

### Changed

- `_cooley_tukey()` now dispatches to NKI butterfly kernel when running on Trainium.
- `_nki_complex_gemm()` and `_nki_complex_mask()` call real NKI kernels instead of PyTorch fallback.

## [0.2.0] - 2026-04-11

### Added

- `istft()` — inverse STFT with overlap-add reconstruction, matching `torch.istft` signature.
- `fftn()` and `ifftn()` — N-dimensional FFT along arbitrary dimensions.
- GitHub Actions CI (Python 3.10, 3.11, 3.12).
- Neuron hardware CI workflow (manual `workflow_dispatch` scaffold).
- Issue and PR templates.

### Changed

- `fft2()` now delegates to `fftn()` internally.
- `neuron-complex-ops` repo archived with pointer to trnfft.

## [0.1.0] - 2026-04-11

### Added

- `ComplexTensor` class with split real/imaginary representation and full arithmetic operator support.
- Complex matrix multiplication decomposed into four real matmuls.
- 1D FFT and IFFT using iterative Cooley-Tukey radix-2 (decimation-in-time).
- Bluestein's chirp-z algorithm for arbitrary-size transforms.
- `rfft` and `irfft` for real-valued input signals.
- 2D FFT (`fft2`) via sequential 1D transforms along each axis.
- STFT matching `torch.stft` signature with `unfold`-based framing.
- Complex neural network layers: `ComplexLinear`, `ComplexConv1d`, `ComplexBatchNorm1d`, `ComplexModReLU`.
- NKI dispatch layer with `auto`, `pytorch`, and `nki` backend selection.
- FFT plan caching (FFTW-style) for repeated transforms of the same size.
- NKI butterfly and GEMM kernel stubs (scaffolded for on-hardware validation).
- Speech enhancement example using complex ideal ratio mask (cIRM).
- 83 tests covering arithmetic, FFT correctness, STFT, NN layers, and gradients.

[Unreleased]: https://github.com/trnsci/trnfft/compare/v0.10.1...HEAD
[0.10.1]: https://github.com/trnsci/trnfft/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/trnsci/trnfft/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/trnsci/trnfft/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/trnsci/trnfft/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/trnsci/trnfft/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/trnsci/trnfft/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/trnsci/trnfft/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/trnsci/trnfft/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/trnsci/trnfft/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/trnsci/trnfft/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/trnsci/trnfft/releases/tag/v0.1.0
