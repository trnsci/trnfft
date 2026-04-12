# trnfft

FFT and complex-valued tensor operations for AWS Trainium via NKI.
Part of the trn-* scientific computing suite by Playground Logic.

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
  3-FFT Bluestein chain in float32. For applications needing higher precision,
  run on float64 input (`x.double()`) — the implementation supports it. NKI
  kernels (FP32-only on Trainium) will need iterative refinement or Kahan
  summation for large arbitrary-size transforms.

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

This repo is part of the `trn*` suite under Playground Logic:
- `trnfft` — FFT + complex ops (this repo)
- `trnblas` — BLAS operations (https://github.com/scttfrdmn/trnblas)
- `trnrand` — Random number generation (https://github.com/trnsci/trnrand)
- `trnsolver` — Linear solvers, eigendecomposition (planned)

All repos: Python/NKI, Apache 2.0, Playground Logic.
