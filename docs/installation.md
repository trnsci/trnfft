# Installation

## Basic install

```bash
pip install trnfft
```

## With Neuron hardware support

```bash
pip install trnfft[neuron]
```

This installs `neuronxcc` and `torch-neuronx` for NKI kernel acceleration on trn1/trn2 instances.

## Development install

```bash
git clone https://github.com/scttfrdmn/trnfft.git
cd trnfft
pip install -e ".[dev]"
pytest tests/ -v
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- NumPy >= 1.24
- neuronxcc >= 2.15 (optional, for Trainium hardware)
