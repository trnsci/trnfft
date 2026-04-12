# Installation

## Basic install

```bash
pip install trnfft
```

## With Neuron hardware support

On a Trainium/Inferentia instance, install into the AMI's pre-built Neuron venv (which already contains `neuronxcc`, since it's not on public PyPI):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install trnfft[dev]
```

The `trnfft[neuron]` extra is only useful when building a custom Neuron environment from scratch.

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
- neuronxcc >= 2.24 (optional, for Trainium hardware)
- torch-neuronx >= 2.9 (optional, for Trainium hardware)

## Hardware compatibility

The NKI kernels in trnfft are validated against this stack:

| Component | Version |
|-----------|---------|
| Neuron SDK (`neuronxcc`) | **2.24.5133.0** (or later 2.24.x) |
| Deep Learning AMI | **Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)** — 20260410 or later |
| Pre-built venv on AMI | `/opt/aws_neuronx_venv_pytorch_2_9` |
| Instance types | `trn1.*`, `trn2.*`, `inf2.*` |
| Python on AMI | 3.12 |

Older Neuron SDKs (< 2.24) used a different `nisa.nc_matmul` calling convention and tile layout requirements; the kernels here will not compile against them. Use the AMI listed above (or its successor in the same major series) for guaranteed compatibility.

If you must use a different SDK version, set `trnfft.set_backend("pytorch")` to skip NKI dispatch entirely. CPU/PyTorch correctness is preserved across all platforms.

For a pre-built CI instance, see [AWS Setup](aws_setup.md) and the Terraform module in `infra/terraform/`.
