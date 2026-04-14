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
git clone https://github.com/trnsci/trnfft.git
cd trnfft
pip install -e ".[dev]"
pytest tests/ -v
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- NumPy >= 1.24
- nki >= 0.3.0 (optional, for Trainium hardware or CPU simulator; Neuron SDK 2.29+)
- torch-neuronx >= 2.9 (optional, for Trainium hardware)

## Hardware compatibility

The NKI kernels in trnfft are validated against this stack:

| Component | Version |
|-----------|---------|
| Neuron SDK | **2.29+** (ships NKI 0.3.0 Stable) |
| `nki` package | **>=0.3.0** |
| Deep Learning AMI | **Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)** with SDK 2.29 bundle |
| Pre-built venv on AMI | `/opt/aws_neuronx_venv_pytorch_2_9` |
| Instance types | `trn1.*`, `trn2.*`, `inf2.*` |
| Python on AMI | 3.12 |

Older Neuron SDKs (< 2.29) used `neuronxcc.nki` namespace and a pre-kwargs `nisa.nc_matmul` calling convention; kernels on 0.12.0+ will not compile against them. If terraform is managing your instance, `terraform apply` auto-picks the latest matching DLAMI. For correctness iteration without hardware, set `TRNFFT_USE_SIMULATOR=1` (see [`docs/api/nki.md`](api/nki.md)).

If you must use a different SDK version, set `trnfft.set_backend("pytorch")` to skip NKI dispatch entirely. CPU/PyTorch correctness is preserved across all platforms.

For a pre-built CI instance, see [AWS Setup](aws_setup.md) and the Terraform module in `infra/terraform/`.
