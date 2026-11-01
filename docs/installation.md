# Installation

## Basic install

```bash
pip install trnfft
```

## With Neuron hardware support

On a Trainium/Inferentia instance, install into the AMI's pre-built Neuron venv (which already contains `nki` and `torch-neuronx`, since they are not on public PyPI):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install trnfft[dev]
```

To build a Neuron environment from scratch, use the extra that matches the instance:

```bash
# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+)
pip install "trnfft[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trnfft[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

`[torchneuron]` pins `torch>=2.14`, `torch-neuronx>=2.14` and `nki>=0.6`
(torch-neuronx 2.14 requires Python ≥ 3.12). `[neuron]` is the XLA lineage
(`torch-neuronx>=2.9`, `nki>=0.3.0`).

### Two Neuron PyTorch lineages

`torch-neuronx` ships in two lineages, and trnfft dispatches its NKI kernels
on whichever one is installed:

| Lineage | Instances | Stack | Kernel dispatch |
|---|---|---|---|
| TorchNeuron (native) | Trn2, Trn3 | Neuron SDK 2.33+, Python ≥ 3.12, `torch` 2.14, `torch-neuronx` 2.14.* | Each kernel is a PyTorch custom op (`torch.ops.trnfft.complex_gemm`, `stockham_radix4_stage`, ...) registered with `torch_neuronx.nki_op`; tensors live on `device="neuron"`. The kernel ops also work inside `torch.compile(backend="neuron")`. Gradients use `torch.library.register_autograd`. |
| PyTorch/XLA | Trn1 (also Inf2) | `torch-neuronx` 2.9 lineage, `torch_xla` | Unchanged: kernels run on the `torch_xla` device, gradients through `torch.autograd.Function`. |

Selection is automatic, on the first kernel launch:

1. **native**: `torch_neuronx` exports `nki_op` and a `neuron` device is
   available (`torch_neuronx.device_count() > 0`). This path never imports
   `torch_xla`.
2. **xla**: `torch_xla` is installed.
3. otherwise there is no device path. Without `nki`, trnfft uses its PyTorch
   implementation; with `nki` but no device, use `set_backend("pytorch")` or
   `TRNFFT_USE_SIMULATOR=1` (unchanged from earlier releases).

The `trnfft::` custom ops are registered only when the native path is
selected, so importing trnfft on CPU or on the XLA lineage has no TorchNeuron
side effects. Both lineages publish a package named `torch-neuronx`, so pick
the extra (or Deep Learning AMI Neuron) that matches the instance.

To force a path, set `TRNSCI_NKI_DISPATCH` before the first launch. The
variable is shared by every trnsci library:

| Value | Effect |
|---|---|
| `auto` (default) | The order above. |
| `native` | Use TorchNeuron; raise `RuntimeError` if no `neuron` device is available. |
| `xla` | Use torch_xla even if TorchNeuron is present; raise `RuntimeError` if torch_xla is missing. |
| `cpu` | No device path (a kernel launch fails as it would without either lineage). |

`TRNSCI_XLA_DEPRECATION=1` makes the XLA path emit one `DeprecationWarning`
when it is selected. It is off by default; the suite turns it on when
TorchNeuron reaches general availability.

### Editable installs and the NKI kernel cache

TorchNeuron enables the NKI trace cache by default
(`NKI_ENABLE_TRACE_CACHE=1`), which persists compiled kernels across
processes. The cache only versions source that belongs to an installed wheel
(a `site-packages` tree with `.dist-info`); an editable install is not
tracked and can get a stale kernel after you edit a kernel. When developing
with `pip install -e`, turn the cache off before starting Python:

```bash
export NKI_ENABLE_TRACE_CACHE=0
```

The test suite keeps the cache on, as users run it, but gives each session
a fresh, empty cache directory (`NKI_TRACE_CACHE_URL`), so edits never hit
a stale entry.

The cache also ignores constants captured in a kernel's closure
([aws-neuron/aws-neuron-sdk#1454](https://github.com/aws-neuron/aws-neuron-sdk/issues/1454));
trnfft's kernels take every specialising value as an argument
(`tests/test_trace_cache_safety.py`). Set `NKI_ENABLE_TRACE_CACHE=0` if
you write NKI kernels that a factory specialises through a closure.

See the TorchNeuron
[environment variables](https://github.com/aws-neuron/torch-neuronx/blob/main/docs/environment_variables.md)
reference.

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
- torch-neuronx >= 2.9 (optional, for Trainium hardware; the native TorchNeuron path needs torch-neuronx 2.14.* on Python >= 3.12)

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
