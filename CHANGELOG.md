# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **The six-step's first pass at N = 2^13 reads x in place (#97).** It no
  longer transposes x and runs the `load_transpose2d` DFT-GEMM. A new kernel,
  `_complex_gemm_tn_kernel`, loads the (B, N2, N1) view of x with plain
  `nl.load`s. Outputs are bit-identical.
  - Cause of the 1024 × 2^13 anomaly in #96: on Trn2 (SDK 2.33) the runtime
    rebuilds a transposed load's IO DMA rings whenever the input address
    changes (`neuron-explorer` system trace: `kbl_exec_pre` 2–9 ms and
    4.7 MB of ring descriptors per launch, against ~12 µs otherwise). Above
    32768 rows the pass is split into launches on different row slices, so
    every launch paid it. In four 1024 × 2^13 calls, 16 launches rebuilt
    their rings before this change and none did after it.
  - Device input, warm medians, ms (logical core 1, host threads on CPUs 3,9):

    | B × 2^13 | before | after |
    |---|---|---|
    | 1 | 1.69 | 1.51 |
    | 16 | 1.69 | 1.46 |
    | 256 | 3.48 | 3.41 |
    | 512 | 13.3 | 5.83 |
    | 1024 | 18.5 | 10.6 |
    | 2048 | 77.6 | 20.1 |
    | 4096 | 156 | 40.4 |

  - `TRNFFT_SIX_STEP_TN=0` restores the old pass. `_fft_via_gemm` (the
    default for N < 512) still uses `load_transpose2d`, so it still pays
    the rebuild once whenever its input is at a new address (#97).

- **Large host results come from the caching host allocator**, as in
  trnblas#64, trnrand#70 and trntensor#57. On the native TorchNeuron path,
  a device-to-host copy into fresh pageable memory runs at about 4 GB/s. A
  copy into `torch.empty(..., pin_memory=True)` runs at 13–14 GB/s. An FFT
  result has two planes (real and imaginary), so it makes two such copies.
  - Every host-output exit uses it: `_launch`, the fused four-step, the
    six-step, the Stockham and butterfly drivers, the batch split across
    cores (each shard now writes into its rows of one output) and the
    host-staged multi-core six-step.
  - Planes of 1 MiB and up are pinned, and only when
    `torch.accelerator.current_accelerator().type == "neuron"`. A CUDA build
    with no GPU raises on `pin_memory=True`.
  - `TRNFFT_PINNED_OUTPUT=0` turns it off. Each call still returns a fresh
    tensor. Cached blocks are rounded up to a power of two and stay cached
    until `torch.accelerator.empty_host_cache()`.
  - Host input and host output, warm medians in ms. Outputs are bitwise equal
    to the pageable ones in every case, forward and inverse
    (`scripts/bench_pinned_output.py`):

    | B × N | pageable | pinned | device-resident |
    |---|---|---|---|
    | 16384 × 4096 | 190 | **88.0** | 10.6 |
    | 16384 × 1024 | 48.2 | **23.4** | 3.54 |
    | 4096 × 4096 | 48.7 | **22.3** | 2.39 |
    | 1024 × 4096 | 13.3 | **6.05** | 0.82 |
    | 1024 × 2^13 | 33.3 | **21.0** | 10.7 |
    | 256 × 2^16 | 59.9 | **35.0** | 15.0 |
    | 16 × 2^20 | 54.4 | **28.4** | 8.50 |
    | 4 × 2^22 | 53.6 | **27.9** | 7.96 |
    | 1 × 2^22 | 14.4 | **8.31** | 3.01 |
    | 1 × 2^24 | 54.2 | **29.1** | 9.30 |
    | 1 × 2^25 | 142 | **94.2** | 54.2 |
    | 1 × 2^26 | 308 | **207** | 129 |

    Results of 8 MiB per plane and less are unchanged within noise
    (1 × 2^13 to 2^20, 16 × 2^13, 256 × 2^13, 64 × 2^14 and 16 × 2^16). The
    inverse gains the same: 16384 × 4096 goes from 181 to 87.7 ms.

- **The six-step FFT is the default for power-of-2 FP32 N >= 8192 (#94).**
  "fast" and "kahan" take it in place of the row-chunked Stockham (fast) and
  compensated butterfly (kahan) paths. That includes `ifft`, `rfft`/`irfft`,
  `fft2`/`fftn` along a long axis, and Bluestein sizes that pad to 8192 or
  more, since they all go through `fft_core`.
  - `TRNFFT_SIX_STEP=0` restores the old path, and `=1` forces the six-step
    for every N above 4096. `TRNFFT_SIX_STEP_MIN` moves the threshold.
  - The six-step won every case measured from 2^13 up, on warm and first
    calls, from device and host input. The smallest gain is 5× at 1 × 2^13
    from the host; from device input it is 9.7×. Error is 11–17× lower.
  - Trn2, one logical core, SDK 2.33. Times are warm medians in ms, and first
    calls in s, each in a fresh process with an empty compile cache. "rel" is
    relative L2 error against FP64 `torch.fft`
    (`scripts/bench_six_step_crossover.py`):

    | B × N | old, device | old, host | old, first call | old rel | six-step, device | six-step, host | six-step, first call | six-step rel |
    |---|---|---|---|---|---|---|---|---|
    | 1 × 2^13 | 12.9 | 11.7 | 9.5 | 1.7e-6 | **1.33** | 2.30 | 6.3 | 1.5e-7 |
    | 1 × 2^14 | 12.9 | 13.2 | 9.5 | 2.1e-6 | **1.66** | 2.07 | 4.4 | 1.4e-7 |
    | 1 × 2^15 | 14.7 | 14.6 | 8.9 | 2.4e-6 | **1.60** | 2.10 | 6.1 | 1.5e-7 |
    | 1 × 2^16 | 16.6 | 16.2 | 12.6 | 2.1e-6 | **1.93** | 2.48 | 4.6 | 1.6e-7 |
    | 1 × 2^18 | 47.9 | 50.7 | 11.1 | 2.9e-6 | **1.67** | 2.10 | 4.6 | 1.8e-7 |
    | 1 × 2^20 | 121 | 112 | 82 | 2.9e-6 | **2.02** | 3.25 | 5.2 | 2.1e-7 |
    | 1 × 2^22 | 698 | 663 | 272 | 2.9e-6 | **3.11** | 15.6 | 6.6 | 2.2e-7 |
    | 1 × 2^24 | 1625 | 1977 | 163 | 3.7e-6 | **9.34** | 55.9 | 14.0 | 2.3e-7 |
    | 1 × 2^25 | 5913 | 5770 | 610 | 3.3e-6 | **82.1** | 190 | 47.6 | 2.3e-7 |
    | 1 × 2^26 | 9577 | 9843 | 627 | 3.7e-6 | **130** | 311 | 50.6 | 2.2e-7 |
    | 16 × 2^13 | 24.4 | 29.9 | 13.8 | 1.7e-6 | **1.71** | 1.89 | 6.6 | 1.5e-7 |
    | 256 × 2^13 | 264 | 244 | 135 | 1.7e-6 | **3.34** | 6.57 | 10.5 | 1.5e-7 |
    | 1024 × 2^13 | 844 | 874 | 541 | 1.7e-6 | **41.7** | 66.9 | 18.4 | 1.5e-7 |
    | 64 × 2^14 | 87.7 | 83.4 | 81.6 | 2.1e-6 | **2.47** | 4.58 | 5.8 | 1.4e-7 |
    | 16 × 2^16 | 114 | 108 | 81.4 | 2.1e-6 | **2.08** | 3.55 | 5.7 | 1.6e-7 |
    | 256 × 2^16 | 2304 | 2264 | 577 | 2.1e-6 | **15.3** | 69.0 | 26.6 | 1.6e-7 |
    | 16 × 2^20 | 1887 | 1817 | 588 | 2.9e-6 | **8.50** | 57.2 | 17.6 | 2.1e-7 |
    | 4 × 2^22 | 2800 | 2788 | 592 | 2.9e-6 | **8.28** | 55.2 | 12.9 | 2.2e-7 |

  - The inverse matches: 1 × 2^14 / 2^20 / 2^24 take 11.4 / 104 / 1788 ms on
    the old path and 1.85 / 2.04 / 9.28 ms on the six-step, at the same errors.
  - "kahan" on the six-step: 1.67 ms at 2^16, 1.51 ms at 2^20 and 1.84 ms at
    16 × 2^16, with errors of 1.6e-7 to 2.1e-7. The old path (the compensated
    butterfly) had not finished its first call at 2^16 after 19 min, so it was
    not run at the larger sizes.
  - 1024 × 2^13 was 12× slower than 256 × 2^13 for 4× the data. #97 has
    the cause and the fix (below).
  - The first calls here are longer than the 8–20 s reported in #93 (48–51 s
    at 2^25 and 2^26). That run reused compiled kernels across cases; these
    runs did not.

- **The fused four-step path moves a host batch to the device once (#94).**
  `_fft_via_four_step_fused` launched on host slices of 4096 rows, so
  `_launch` copied each slice in and each result out, and the results were
  concatenated on the host. It now moves the batch in once, runs the launches
  on device slices, and returns the result once, as the six-step does.
  Outputs are bit-identical. Host in and out, warm ms:

    | B × N | before | after | caller moves the batch (`.to` / `.cpu()`) | device-resident |
    |---|---|---|---|---|
    | 16384 × 4096 | 348 / 353 | **185 / 185** | 186–230 | 11.1 |
    | 16384 × 1024 | 90 / 93 | **56 / 53** | 49–56 | 3.7 |
    | 4096 × 4096 | 57 | **49** | 49–50 | 2.7 |
    | 1024 × 4096 (one launch) | 13.5 | 13.5 | 13.8–14.0 | 0.8–1.1 |

  - The per-slice copies were not the cost on their own. Four 64 MB
    host-to-device copies took 43 ms against 40 ms for one 256 MB copy, and
    device-to-host took 173 against 167 ms. Concatenating the slices on the
    host took 136 ms on 2 CPUs. The copy back after each launch probably also
    kept the launches from overlapping; that part is inferred, not profiled.

### Added

- **Six-step FFT for large N (`TRNFFT_SIX_STEP`, #54, #93; default since #94).**
  Power-of-2 FP32 transforms above N=4096 can run as N = N1·N2: N1 row FFTs
  of length N2, a twiddle multiply, then N2 row FFTs of length N1, with three transposes, all
  on the device. Each row pass is one batched transform on the batch-fused
  four-step kernel (#82). The default path runs 8–13 row-chunked Stockham stages
  (#83).
  - The split is balanced up to 2^24. Above that, N1 = 4096 and the N2 pass
    recurses.
  - Passes from length 128 up use the fused kernel. `TRNFFT_SIX_STEP_FUSED_MIN`
    sets this length; the normal dispatch starts the fused kernel at 512.
  - Passes below that length use the single DFT-GEMM, split into launches of at
    most 2^15 rows (`TRNFFT_SIX_STEP_GEMM_MAX_ROWS`).
  - A host input moves to the device once.
  - Trn2, one logical core, warm medians in ms. Device-resident input unless
    noted. The CPU column is `torch.fft` on 2 pinned threads:

    | B × N | default | six-step | six-step, host in/out | CPU |
    |---|---|---|---|---|
    | 1 × 2^20 | 115 | 1.72 | 3.42 | 2.55 |
    | 1 × 2^22 | 616 | 3.06 | 8.68 | 22.9 |
    | 1 × 2^24 | 2006 | 9.15 | 57.1 | 100 |
    | 1 × 2^25 | – | 98.6 | 213 | 208 |
    | 1 × 2^26 | – | 130 | 355 | 447 |
    | 16 × 2^20 | 1869 | 8.52 | 56.4 | 64.4 |
    | 256 × 2^16 | 2105 | 15.2 | 64.3 | 46.9 |

  - Relative L2 error against FP64 is 1.6e-7 to 2.3e-7 with the six-step,
    against 2.1e-6 to 3.7e-6 on the default path. CPU measures 1.5e-7 to 2.5e-7.
  - First calls took 8–20 s from 2^20 to 2^26. The default path's first call
    took 203 s at 2^24 and 247 s at 2^26 (#83).
  - The default dispatch was unchanged when the variable was unset. It is
    now the default from N=8192 (#94, above).
- `scripts/bench_multicore.py`: the measurements above and below. It covers the
  default and six-step paths on one core, `torch.fft` on CPU, per-core resident
  shards, the host batch split, process per core, and the six-step split across
  cores.

### Changed

- **`set_multicore(True)` now runs host-batch shards on separate NeuronCores
  (#54, #93).** On the native TorchNeuron path, shard *i* of a host batch goes
  to `neuron:i` and gets its own host thread.
  - The first call for a shard layout runs the shards in turn, because NKI
    tracing is not thread-safe.
  - Native launches are serialised behind a lock. A launch only enqueues work,
    so the cores still run concurrently.
  - A batch already on a NeuronCore is no longer split. Reaching the other
    cores through the host cost 15–20× the single-core transform.
  - Before this change, the shards ran in turn on one device.
- Trn2 measurements (4 logical cores, 2 host CPUs). The host link sets the
  time, not the cores:
  - Host-to-device copies run at about 12 GB/s whether 1, 2 or 4 cores copy
    at once.
  - Device-to-host runs at 3.4 GB/s for 64 MB tensors and about 10 GB/s as
    four 16 MB copies.
  - The threaded host split is at most 1.5× faster than one core. It is
    slower than `torch.fft` on the same 2 CPUs in every case measured.
  - One process per core does better: 213 → 120 ms at B=16384, N=4096,
    against 163 ms on CPU. It is not part of the library.
  - Shards the caller places on `neuron:0..3`, with `trnfft.fft` called on
    each from one thread, scale on compute alone: 10.9 → 6.3 → 4.9 ms at
    B=16384, N=4096 on 1 / 2 / 4 cores. This already worked before this
    change.
- A six-step single transform with both passes split across cores
  (`multicore._six_step_multicore`, host-staged) is slower than one core:
  169 ms on 4 cores against 9.15 ms on one, at N=2^24. It is kept for the
  benchmark and is not wired into dispatch.

## [0.24.0] - 2026-10-30

Native TorchNeuron dispatch (Trn2/Trn3) alongside the PyTorch/XLA path (Trn1, maintenance); validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0) — see trnsci/trnsci#35.

Headline performance: batch-fused four-step FFT kernel, one launch per batch; N=4096 B=1024 goes from 10.1 to 0.93 ms on Trn2 (#91).

### Added

- **Native PyTorch dispatch on TorchNeuron, alongside XLA (#80).** On
  TorchNeuron (Neuron SDK 2.33+, `torch-neuronx` 2.14, PyTorch 2.14,
  Trn2/Trn3) every NKI kernel is registered as a PyTorch custom op with
  `torch_neuronx.nki_op` and dispatched on `device="neuron"`, with no
  `torch_xla` import. Ops: `trnfft::complex_gemm`, `complex_gemm_bf16`,
  `complex_linear`, `complex_mul`, `butterfly_stage`, `butterfly_stage_kahan`,
  `stockham_radix4_stage`, `stockham_radix8_w8`, plus the composite
  `trnfft::fft`. Complex values stay real/imag pairs. Kernel bodies are
  unchanged. Earlier Trn2 results ran through a test-only `torch_xla`
  stand-in; this is the real native path. Trn2 native suite: 62 passed,
  1 xfailed (composite `trnfft::fft` under `torch.compile`); Trn1 XLA shows
  no regression against main.
  - The dispatch path is detected on the first kernel launch: native when
    `torch_neuronx` exports `nki_op` and a `neuron` device is available, XLA
    when `torch_xla` is installed, otherwise none. The XLA path (Trn1) is
    unchanged, as are the PyTorch fallback and `TRNFFT_USE_SIMULATOR`.
  - One launch helper, `dispatch._launch`, carries every kernel launch: the
    host wrappers, `complex_gemm_bf16`, the autograd forwards, the butterfly
    driver, and the row-chunked Stockham stages (`_launch_stockham_stage`,
    #83), which keep their operands on the device between stages. The
    single-GEMM DFT and the fused four-step (#82) go through it via
    `complex_gemm` / `complex_mask_apply`.
  - Autograd on the native path uses `torch.library.register_autograd` on
    `complex_gemm`, `complex_linear`, `complex_mul` and `trnfft::fft` (the
    `fft_autograd` adjoint, IFFT·n / FFT/n). The formulas are shared with the
    `torch.autograd.Function` wrappers, which the XLA path keeps.
  - Custom ops are registered lazily (`trnfft/nki/_native.py` is imported only
    when the native path is selected), so CPU-only installs and the XLA
    lineage see no import-time side effects.
  - `TRNSCI_NKI_DISPATCH=auto|native|xla|cpu` forces the dispatch path (a
    suite-wide variable shared by all trnsci libraries). Forcing an
    unavailable path raises instead of falling back.
  - `TRNSCI_XLA_DEPRECATION=1` makes the XLA path emit one
    `DeprecationWarning`. Off by default (trnsci#35).
  - New `[torchneuron]` extra (`torch>=2.14`, `torch-neuronx>=2.14`,
    `nki>=0.6`); `[neuron]` stays the PyTorch/XLA lineage.
  - Tests: `tests/test_native_dispatch.py` (CPU: path selection, launch
    routing, lazy registration, schemas, fake shapes, the deprecation hook,
    and register_autograd gradients against torch references);
    `tests/test_nki_native.py` (`@pytest.mark.neuron`, native only: each op
    against `nki.simulate`, end-to-end FFT, gradients, `torch.compile`).
    `conftest.py` sets `NKI_ENABLE_TRACE_CACHE=0` by default.
  - Tests: new `slow` marker; slow tests are skipped unless `--run-slow` or
    `-m slow` is given. `TestKahanCharacterization` at N=8193 (about 19 min on
    Trn2) is marked slow so the default `-m neuron` run stays short.

### Changed

- **Batched FFT: one fused four-step kernel launch per batch (#82 follow-up).**
  New `four_step_fft_kernel` (`trnfft/nki/four_step.py`, native op
  `trnfft::four_step_fft`) runs the whole four-step transform (GEMM, twiddle,
  GEMM) for a group of rows in SBUF/PSUM. It reads the batch once and writes it
  once, with no transposes. The unfused driver needed 2 GEMM launches, 1
  twiddle launch and 6 transpose copies, each over the whole batch. Default
  for FP32 power-of-2 512 ≤ N ≤ 4096. N ≤ 256 keeps the single DFT-GEMM.
  - Where the time went: profiling on Trn2 at N=4096 put 78% of B=1024
    (10.9 of 13.9 ms) inside the two `_complex_gemm_kernel` launches, which
    re-transpose A through DMA on each of 512 M-tiles with a 64-wide moving
    operand. At B=16 the cost was the ~0.3 ms dispatch floor times ~9 device
    ops. Host Python was about 1.6 ms per call, overlapped with the device.
  - Kernel layout: each row is read as an `(N1, N2)` matrix whose partition
    axis is already step 1's contraction axis. P = 128/N2 rows are packed per
    stationary operand, and step 2 uses a block-diagonal W2, so all 128 PE
    columns are used. X_re and X_im are stacked on the contraction axis
    against `[[W1r, W1i], [-W1i, W1r]]`, so complex step 1 is one K=128
    matmul. Trn2 at N=4096, B=1024: one row per stationary 3.13 ms, packed
    rows 1.31 ms, two-matmul step 1 0.88 ms, one K=128 matmul 0.79 ms
    (kernel only). Splitting the twiddle across GpSimd/Scalar made it slower
    and was dropped.
  - Trn2 (SDK 2.33 / NKI 0.7.0), device-resident warm median ms, before →
    after (2-thread CPU `torch.fft.fft` in brackets): N=4096 B=1024 10.11 →
    0.93 [10.97]; B=128 3.27 → 0.44 [0.38]; N=2048 B=1024 2.67 → 0.62
    [1.33]; N=1024 B=1024 0.83 → 0.54 [0.58]. Rel err vs FP64 is unchanged
    at 2048/4096 (1.5e-7/1.6e-7) and lower at 512/1024 (1.22e-7/1.41e-7,
    was 1.36e-7/1.54e-7). Results are not bit-identical to the unfused
    driver (step 1 accumulates K=2·N1 in one pass): max abs difference is
    6e-5 on outputs of order 10².
  - It also removes an unfused-driver limit: at N=4096 the fused-twiddle
    `_complex_mul_kernel` asserted for any B > 16 that is not a multiple of
    16 (e.g. B=38). The fused kernel takes any B.
  - Knobs: `TRNFFT_FOUR_STEP_BATCH_FUSED=0` restores the unfused dispatch.
    `TRNFFT_FOUR_STEP_FUSED_MIN` (default 512) sets the lower N bound.
    `TRNFFT_FOUR_STEP_FUSED_MAX_ROWS` (default 4096) splits larger batches
    into several launches, because first-call compile time grows with B
    (3 s at B=1024, 17 s at B=8192).
  - `scripts/bench_batched_fft.py` reproduces the table.

## [0.23.0] - 2026-10-28

Validated on Trn2 (trn2.3xlarge, Neuron SDK 2.33.0, native TorchNeuron device) — see trnsci/trnsci#35.

### Changed

- **NKI FFT dispatch for 256 < N ≤ 4096 rebuilt around GEMM launches, defaults
  chosen from a Trn2 A/B** (`trnfft/fft_core.py`, #82). Spike B7 on Trn2 had
  batch-32 FFT at 7.1 ms (N=1024) and 10.5 ms (N=4096), against 0.53 ms at
  N=256. Dispatch counts per warm call, from fake kernels on CPU with `meta`
  standing in for the device, explain it. N=1024 ran 4 Stockham kernels,
  26 inter-stage permute/twiddle device ops and 10 host-to-device twiddle
  copies, and rebuilt the twiddles on the host (about 40 ops) every call.
  N=4096 ran 4 kernels, 38 device ops and 10 copies.
  - **Single DFT-GEMM default threshold raised 256 → 1024.** W is cached on
    the device: no per-call N×N FP64 build or host-to-device copy. For
    power-of-2 N the inverse 1/N is folded into W, an exact scale.
  - **New four-step FFT (`_fft_via_four_step`) for 1024 < N ≤ 4096.** N = N1·N2
    (64×64 at N=4096). It runs GEMM(x, W_N1), a twiddle multiply, then
    GEMM(·, W_N2): 2 GEMM launches, each accumulating K ≤ 64 terms.
  - **The four-step twiddle is fused** into one `_complex_mul_kernel` launch.
  - Trn2 (SDK 2.33 / NKI 0.7.0), batch 32, device-resident warm median ms.
    Columns: pre-#82 Stockham / four-step / four-step with fused twiddle /
    single GEMM.
    - N=512: 5.45 / 1.67 / 1.23 / **0.26**
    - N=1024: 7.56 / 2.22 / 2.04 / **0.31**
    - N=2048: 8.22 / 1.58 / **1.20** / 1.08
    - N=4096: 10.25 / 2.43 / **2.18** / 2.36

    The new defaults are the bold entries. At N=2048 the single GEMM is
    slightly faster, but four-step is the default there because W would be
    2048×2048.
  - Accuracy on Trn2 (rel err vs FP64): single GEMM 1.35e-7 at N=512 and
    1.54e-7 at N=1024; four-step 1.2e-7 to 1.6e-7 at N=64–4096, the same with
    the fused twiddle. CPU emulation of Stockham gave 1.3–2.0e-6.
  - The old comment blaming the 256 cap on "2.2% rel err at N=1024" was wrong.
    That error came from the pre-#81 FP32 twiddle build, not from PSUM
    accumulation, and the comment has been replaced.
  - "kahan" stays on the butterfly path. The bf16 / bf16_refined / ozaki /
    ozaki_hq GEMM modes keep their characterised N ≤ 256 range
    (`_LOWP_GEMM_MAX`); above it they fall through to the FP32 GEMM paths.
- **Env knobs** (read at import):
  - `TRNFFT_DFT_GEMM_THRESHOLD` (default 1024). 0 disables every GEMM path.
  - `TRNFFT_FOUR_STEP_MAX` (default 4096). 0 restores the pre-#82 Stockham
    dispatch above the threshold.
  - `TRNFFT_FOUR_STEP_FUSED_TWIDDLE` (default 1). 0 uses 6 elementwise device
    ops instead of the fused kernel.

### Fixed

- **`precision="bf16_refined"` gave no refinement** (`trnfft/fft_core.py`,
  #86). Trn2 measured 1.95e-3 / 2.03e-3 / 2.08e-3 normwise relative error at
  N=64/128/256, the same as plain `"bf16"`, against a 1e-5 promise. CPU
  emulation of the BF16 kernel reproduces those numbers to 3 digits, so the
  fault was in the algorithm, not in the device path or the dispatch. The
  #82 routing still reaches `_fft_iterative_refinement` for N ≤ 256. The cause
  was that the IR-1 step reconstructed x from X̂ with the BF16 inverse DFT.
  With an approximate operator on both sides, the residual's correction
  cancels the forward BF16 W error but adds the inverse's error, which is the
  same size. The error stays O(u_bf16). The reconstruction now uses the FP32
  DFT-GEMM (`_fft_via_gemm`, W built in FP64). CPU emulation gives 3.6e-6
  (N=2) to 4.9e-6 (N=256), forward and inverse, max over 20 seeds, which is
  2× under the unchanged 1e-5 bound. More steps converge to the FP32 floor
  (2.9e-7 at N=256 with 3 steps). Cost per step is now two BF16 GEMMs plus one
  FP32 GEMM, so the mode is not cheaper than `"fast"`. The docstrings say so.
- **`precision="bf16"` bound restated to 3e-3** (`trnfft/precision.py`, #86).
  The error is BF16 rounding of W and x, about u_bf16 = 2^-9. It grows slowly
  with log N: 1.9e-3 (N=2) to 2.16e-3 (N=256) worst case over 20 seeds on CPU,
  and Trn2 measured 2.06e-3 / 2.09e-3 at N=128/256. The old "≈1e-3 at N=256"
  text, read as a 2e-3 bound, had no headroom over u_bf16.
- New CPU tests (`TestBF16BoundsRegression`) pin both modes against their
  bounds at N=2–256, forward and inverse, and pin the NKI dispatch of
  bf16/bf16_refined for N ≤ `_LOWP_GEMM_MAX`.
- **FFT N ≥ 2^25 failed on Trn2 with "body blob exceeds uint32 offset range"**
  (`trnfft/fft_core.py`, #83). Each Stockham stage launches one kernel over a
  `(B·N/r, r)` operand, and the compiler unrolls the kernel's 128-row tile
  loop, so the kernel body grows with the row count. At N=2^25 the radix-4
  stage took 2^23 rows and BIR emission failed. N ≤ 2^24 passed; the largest
  launches there were 2^21 rows. Each row is an independent r-point butterfly
  group with its own twiddle row, so the radix-4, radix-8 and mixed-radix
  drivers now split any stage above 2^21 rows into launches of at most 2^21
  rows (on 128-row tile boundaries) and concatenate the results. Output is
  bit-identical to a single launch. Env `TRNFFT_MAX_ROWS_PER_LAUNCH` overrides
  the cap and is read per call.

- **`precision="kahan"` is no longer less accurate than `"fast"`** (#82). On
  Trn2 at bbff7be, `TestKahanCharacterization` failed at Bluestein N=997:
  kahan 4.09e-7 against fast 2.62e-7, where the tests require kahan ≤ 1.5×fast.
  The fast path had moved to the single-GEMM and four-step GEMM paths, but
  kahan's inner power-of-2 FFTs stayed on the Kahan butterfly, which
  measures 1.9e-7 to 4.6e-7 on trn1. The GEMM paths are more accurate on
  Trn2 (1.2e-7 to 1.6e-7), so `"kahan"` now uses them up to N=4096, the same
  as `"fast"`. The Kahan butterfly still applies above N=4096, or at any N with
  `_DFT_GEMM_THRESHOLD = 0`. Bluestein keeps the compensated chirp multiplies.
  The butterfly twiddles on the NKI and CPU paths are now built in FP64 and
  cast once.
  - CPU with fake kernels on the NKI dispatch path, using the neuron test's
    max-rel metric at seed 42. Columns: fast / kahan before → kahan after.
    - N=997: 3.37e-7 / 3.55e-7 → 3.44e-7
    - N=1009: 3.21e-7 / 4.09e-7 → 3.80e-7
    - N=8193: 3.66e-6 / 5.47e-7 → 2.59e-7. At this size fast uses Stockham
      and kahan the butterfly.
    - N=4096 (power of 2): 2.46e-7 / 4.25e-7 → 2.46e-7. Before the fix this
      ratio was 1.73×, a latent failure of the same contract.
  - Over 20 seeds the kahan/fast ratio at N=997 and N=1009 now averages
    1.00 and is at most 1.16.

- **Bluestein (non-power-of-2 N) chirp is now built in FP64 with k² reduced
  mod 2N** (`trnfft/fft_core.py`, found in the #82 Trn2 run). The chirp
  `exp(±iπk²/N)` was computed from an FP32 angle of up to about πN, which put
  about N·1e-7 relative error into the chirp. Error was 1.2e-4 at N=1000 on
  Trn2, against 5.1e-7 at N=7; it is the same bug class as #81. CPU results,
  forward and inverse, batch 4:
  - N=1000: 1.17e-4 → 3.7e-7
  - N=4097: 4.4e-4 → 4.0e-7
  - N=7: 4.2e-7 → 1.1e-7

- **`precision="ozaki_hq"` now meets its 1e-6 bound, and `"ozaki"` its 1.6e-5 bound,
  at N ≤ 256** (`trnfft/fft_core.py`, #81). Spike B5 on Trn2 (SDK 2.33) measured
  ozaki_hq at 4.9e-6 / 9.9e-6 / 2.0e-5 and ozaki at 5.9e-6 / 1.0e-5 / 2.0e-5 at
  N=64/128/256, so HQ gave no gain. CPU emulation reproduces those numbers to two
  digits. There were two causes:
  1. The DFT matrix was built as `cos(2π·k·j/N)` in FP32 with no `k·j mod N`
     reduction. Angles up to ~2πN rounded to FP32 give a twiddle error of about
     N·1e-7 relative, which neither split can remove. That error grows linearly
     with N and is the same for both modes.
  2. ozaki_hq split W only 2-way, which leaves a ~u_bf16² residual in W
     (~1.3–1.9e-6 even with exact twiddles).

  New `_dft_matrix_fp64` builds W in FP64 with angles reduced mod N. All DFT-GEMM
  paths now use it (`fast`, `double`, `bf16`, `ozaki`, `ozaki_hq`). Both Ozaki
  modes split W from FP64. ozaki_hq splits W 3-way and uses the triangular
  6-term product (level(W) + level(x) ≤ 2), so it still runs 6 matmuls. CPU
  results: ozaki 3.4e-6–3.8e-6, ozaki_hq 6.8e-8–1.1e-7. The old "≈ 2e-9" ozaki_hq
  claim ignored the FP32 output floor and has been removed. The N=512 B5 row
  never ran Ozaki: both modes dispatch only for N ≤ 256 and fall through to FP32
  Stockham above that. This is now documented. Hardware re-check pending.

- **`complex_gemm_bf16` returns results on the caller's device** (`trnfft/nki/dispatch.py`).
  The hardware path moved inputs to the device but returned the kernel outputs without
  moving them back, so `_fft_via_ozaki_hq` mixed device and CPU tensors
  (`RuntimeError: input tensor is on cpu device, expected neuron`; all four
  `TestOzakiHQCharacterization` failures on Trn2 native, #80). It now uses the shared
  `_to_xla` helper and moves `c_real` / `c_imag` back, matching the autograd wrappers and
  the fft_core drivers. Audit of the other host wrappers in `trnfft/nki/` found no other
  instance.

- **Bluestein (non-power-of-2) FFT keeps every tensor on the caller's device**
  (`trnfft/fft_core.py::_bluestein`, #80 follow-up). `fft`/`ifft` at every
  non-power-of-2 length failed on Trn2 native with `RuntimeError: input tensor is on
  cpu device, expected neuron`. The chirp, the zero-padded `y` and the filter `h` were
  built with CPU-default `torch.arange`/`torch.zeros`, so `x * chirp` mixed a device
  input with a CPU chirp. They are now created on (or moved to) `x.real.device`; `h` is
  built with one vectorized wrap instead of a per-element loop. The same CPU-default
  factory pattern is fixed in the batch padding of the NKI butterfly/Stockham drivers,
  the bit-reversal indices and CPU butterfly twiddles, the PyTorch Stockham reference
  (`trnfft/stockham.py`), the four-step twiddles (`trnfft/nki/multicore.py`), and the
  `irfft`/`stft`/`istft`/`n=` padding buffers and default windows (`trnfft/api.py`).
  CPU-verified (meta-device stand-in); hardware re-check pending.

### Added

- `tests/test_nki_compile.py` (`nki_simulator` marker): runs the NKI frontend
  (trace -> MLIR -> verify) on CPU for every `@nki.jit` kernel in `trnfft/nki/` on trn1
  and trn2. `nki.simulate` does not run the MLIR verifier; this catches verifier errors
  without hardware. All eight kernels verify on NKI 0.7.0.
- CPU regression test for the `complex_gemm_bf16` device-return contract
  (`tests/test_complex.py::TestDispatchDeviceReturn`).
- CPU regression test for non-power-of-2 FFT/IFFT device placement through the NKI
  dispatch path (`tests/test_fft.py::TestBluesteinDeviceReturn`). The `meta` device
  stands in for `neuron` (it also rejects mixing with CPU tensors); fails before the fix.

## [0.22.0] - 2026-05-15

### Added

- **STFT/ISTFT multi-NeuronCore dispatch** (`trnfft/api.py`). `stft()` and `istft()` now
  route the inner FFT batch through `multi_core_fft()` instead of `fft_core()`. When
  `set_multicore(True)` is active, STFT frames — which are independent transforms — are
  distributed across NeuronCores via `_batch_split_fft`. Behaviour is unchanged when
  multicore is disabled (falls back to `fft_core`). No API changes.

  A 16 kHz 1-s waveform with n_fft=512, hop=256 produces ~62 frames; each frame is an
  independent length-512 FFT, making STFT a natural batch workload for data-parallel
  multi-core dispatch.

- `benchmarks/bench_fft.py::TestSTFTMulticore` — hardware benchmark comparing multicore
  vs single-core STFT at n_fft ∈ {128, 256, 512}. Added to `scripts/run_benchmarks.sh`.

- `HAS_COLLECTIVES` capability flag in `trnfft/nki/multicore.py`. Checks for
  `nki.collectives` at import time. Currently False on all available AMIs (SDK 2.29.x).
  Documents the NeuronLink stage-parallel forward path for SDK ≥ 2.30.

  trn2 hardware results (sa-east-1b, SDK 2.29.1, 2026-05-16) — all 23 benchmarks passed.
  Note: `torch_neuronx.DataParallel` is not usable for multi-shape dispatch in SDK 2.29
  (internal structure flattener asserts layout equality across calls). `_batch_split_fft`
  dispatches shards sequentially through NKI-compiled `fft_core`. True cross-core
  parallelism deferred to NeuronLink collectives (SDK ≥ 2.30).

  **trn2 vs trn1 throughput (median µs):**

  | Algorithm | N | trn1 (µs) | trn2 (µs) | trn2/trn1 |
  | --------- | --- | --------- | --------- | --------- |
  | BF16 DFT-GEMM | 64 | 1 179 | 998 | 0.85× |
  | BF16 DFT-GEMM | 256 | 1 300 | 1 092 | 0.84× |
  | Stockham r4 | 64 | 4 322 | 5 986 | 1.39× |
  | Stockham r4 | 1024 | 6 850 | 8 039 | 1.17× |
  | Stockham r4 | 4096 | 8 632 | 9 637 | 1.12× |
  | Stockham r8 | 64 | 4 793 | 5 344 | 1.11× |
  | Stockham r8 | 4096 | 5 917 | 7 397 | 1.25× |
  | Stockham mixed | 1024 | 5 863 | 7 324 | 1.25× |
  | Ozaki (1-level) | 64 | 3 225 | 2 749 | 0.85× |
  | Ozaki (2-level) | 64 | 6 252 | 5 436 | 0.87× |
  | STFT multicore | n_fft=128 | — | 3 943 | — |
  | STFT multicore | n_fft=256 | — | 4 241 | — |
  | STFT multicore | n_fft=512 | — | 13 366 | — |

  trn2 is 10–16% faster than trn1 for DFT-GEMM / Ozaki paths (memory-bandwidth bound).
  trn2 is 11–39% slower for Stockham (kernel-launch overhead dominates at small N;
  trn2's higher clock rate doesn't help if the bottleneck is dispatch latency).

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
