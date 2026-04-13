# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
