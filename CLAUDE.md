# trnfft

FFT and complex-valued tensor operations for AWS Trainium via NKI.
Part of the trnsci scientific computing suite.

Incorporates neuron-complex-ops (https://github.com/scttfrdmn/neuron-complex-ops).

## What This Is

A cuFFT-equivalent for Trainium. Trainium has no native complex dtype and no
FFT library. trnfft fills that gap using split real/imaginary representation,
complex neural network layers, and NKI kernels targeting the Tensor, Vector,
and Scalar engines.

## Relationship to neuron-complex-ops

`neuron-complex-ops` was the original speech-enhancement proof-of-concept.
This project absorbs it:

**Folded in from neuron-complex-ops:**
- `ComplexTensor` with full arithmetic → `trnfft/complex.py`
- NKI dispatch layer (pytorch ↔ nki auto-select) → `trnfft/nki/dispatch.py`
- Complex GEMM kernel (stationary reuse, PSUM accumulation) → `trnfft/nki/dispatch.py`
- Complex mask apply kernel → `trnfft/nki/dispatch.py`
- DFT-matrix-multiply STFT → absorbed into `trnfft/api.py` stft()
- ComplexLinear, ComplexConv1d, ComplexBatchNorm1d, ComplexModReLU → `trnfft/nn.py`
- Speech enhancement example (cIRM training) → `trnfft/examples/`

**trnfft adds on top:**
- Cooley-Tukey / Bluestein FFT algorithms (`trnfft/fft_core.py`)
- Plan-based execution with caching (`trnfft/plan.py`)
- torch.fft-compatible API surface (`trnfft/api.py`)
- NKI butterfly kernels for hardware-accelerated FFT (`trnfft/nki/butterfly.py`)
- Multi-NeuronCore parallelism for large transforms (future)

neuron-complex-ops repo stays public with a pointer to trnfft.

## Architecture

```
trnfft/
├── trnfft/
│   ├── __init__.py          # Public API re-exports
│   ├── api.py               # torch.fft-compatible: fft, ifft, rfft, irfft, fft2, stft
│   ├── complex.py           # ComplexTensor (split real/imag) — from neuron-complex-ops
│   ├── fft_core.py          # Cooley-Tukey + Bluestein algorithms
│   ├── nn.py                # ComplexLinear, ComplexConv1d, ComplexBatchNorm1d, ComplexModReLU
│   ├── plan.py              # FFTPlan with caching
│   ├── nki/
│   │   ├── __init__.py
│   │   ├── dispatch.py      # PyTorch ↔ NKI auto-dispatch + GEMM/mask kernels
│   │   └── butterfly.py     # NKI butterfly kernel stubs (validate on hardware)
├── tests/
│   ├── conftest.py          # Fixtures, hardware detection
│   ├── test_complex.py      # ComplexTensor arithmetic + matmul
│   ├── test_fft.py          # 1D/2D FFT, IFFT, RFFT, Bluestein
│   ├── test_stft.py         # STFT shape, energy, tone detection
│   └── test_nn.py           # Complex neural network layers
├── examples/
│   └── speech_stft.py       # cIRM speech enhancement demo
├── benchmarks/
├── docs/
├── pyproject.toml
├── README.md
├── LICENSE                  # Apache 2.0
└── CLAUDE.md                # This file
```

## Key Design Decisions

1. **Split real/imaginary** — Trainium has no complex dtype. ComplexTensor wraps paired real tensors with arithmetic operators.

2. **Iterative Cooley-Tukey** — Standard radix-2 DIT with bit-reversal. Each butterfly stage vectorized across all groups simultaneously.

3. **Bluestein for arbitrary sizes** — Converts arbitrary-N DFT to circular convolution via three power-of-2 FFTs.

4. **Plan-based execution** — Like FFTW/cuFFT. Plans cached by (size, inverse).

5. **torch.fft-compatible API** — `trnfft.fft(x)` mirrors `torch.fft.fft(x)`.

6. **NKI dispatch** — `set_backend("auto"|"pytorch"|"nki")`. Auto-detects Neuron hardware.

## Known Gaps & Design Notes

- **NKI kernels are stubs.** Both GEMM and butterfly kernels fall back to
  PyTorch even when NKI is available. Next milestone is on-hardware validation
  on a trn2 instance, wiring the actual kernels from neuron-complex-ops.

- **Bluestein FP32 precision.** N>=500 accumulates ~2e-2 error through the
  3-FFT Bluestein chain in float32. Use `trnfft.set_precision("double")` to
  promote Bluestein host math to FP64 (~1e-11 rel error at any N); power-of-2
  Cooley-Tukey is unaffected. `set_precision("kahan")` switches the NKI
  butterfly to a Dekker 2Prod compensated variant for hardware where FP64
  isn't available. See `docs/architecture.md` for the full precision table.

- **ComplexBatchNorm** uses independent real/imag normalization, not the
  covariance-based variant (Trabelsi et al. 2018). Deliberate simplicity —
  the simple form works for cIRM speech enhancement.

## Dependencies

- `torch>=2.1` — tensor operations and CPU fallback
- `numpy>=1.24` — reference FFT for testing
- `neuronxcc` — NKI kernels (optional, only on Neuron hardware)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v               # CPU fallback mode
pytest tests/ -v -m neuron     # On-hardware tests (requires trn instance)
python examples/speech_stft.py --demo  # Quick STFT demo
```

## Naming Convention

This repo is part of the `trn*` suite (all released on PyPI under org `trnsci`):
- `trnfft` — FFT + complex ops (this repo)
- `trnblas` — BLAS Level 1–3 (https://github.com/trnsci/trnblas)
- `trnrand` — Philox / Sobol / Halton RNG (https://github.com/trnsci/trnrand)
- `trnsolver` — Linear solvers + eigendecomposition (https://github.com/trnsci/trnsolver)
- `trnsparse` — Sparse matrix ops (https://github.com/trnsci/trnsparse)
- `trntensor` — Tensor contractions / einsum (https://github.com/trnsci/trntensor)
- `trnsci` — Umbrella meta-package (https://github.com/trnsci/trnsci)

All repos: Python/NKI, Apache 2.0.

## Blog posts

When asked to draft a blog post for this library for the [trnsci blog](https://trnsci.dev/blog/):

1. Read the editorial brief at [`docs/blog/AUTHOR_BRIEF.md`](https://github.com/trnsci/trnsci/blob/main/docs/blog/AUTHOR_BRIEF.md) in the umbrella repo (trnsci/trnsci). It defines voice (authorless, library-as-subject), stance (architecture-first, transparency-always), and the nine required section headings.

2. Find the prompt block for this library in [`BLOG_PROMPTS.md`](https://github.com/trnsci/trnsci/blob/main/BLOG_PROMPTS.md) at the umbrella repo root. It carries library-specific context and suggested architectural angles.

3. Draft the post following the brief. Open a PR against `trnsci/trnsci` at `docs/blog/posts/<YYYY-MM-DD>-<slug>.md`. Scott (suite director) reviews before merge.

The umbrella repo — not this one — owns the blog. Per-library retrospective posts are unsigned; library is the subject, no byline. See the brief for the full set of rules.
