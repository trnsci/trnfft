# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/scttfrdmn/trnfft/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/scttfrdmn/trnfft/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/scttfrdmn/trnfft/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/scttfrdmn/trnfft/releases/tag/v0.1.0
