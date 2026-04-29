"""Multi-NeuronCore parallelism for large FFT transforms.

Partition strategies
--------------------

1. **Data parallelism (batch split)** — For input with batch dimension > 1,
   distribute shards across NeuronCores. No inter-core communication. Speedup
   scales linearly with core count up to min(batch, n_cores).

2. **Stage parallelism (single-transform split)** — A single large FFT split
   across cores by distributing butterfly stages. Requires inter-core all-reduce
   after each stage boundary. Only pays for N > 2^17 on trn1. Not yet
   implemented; single-transform input raises NotImplementedError.

On Trainium hardware with torch_neuronx available, batch splits use
torch_neuronx.DataParallel with a cached compiled model. On CPU (no
torch_neuronx) each shard is processed sequentially — architecturally correct,
no hardware parallelism.

Size thresholds (trn1, 16 NeuronCores, 32 MB SBUF each)
---------------------------------------------------------
Transforms up to N=2^16 fit in one core; stage parallelism is only beneficial
for N > 2^17. These thresholds need measurement and are not enforced here.
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..complex import ComplexTensor

HAS_TORCH_NEURONX = importlib.util.find_spec("torch_neuronx") is not None

# Disabled by default. Enable via TRNFFT_MULTICORE=1 or set_multicore(True).
_use_multicore = os.environ.get("TRNFFT_MULTICORE", "0") == "1"

# Default target NeuronCore count. 0 means "all available" (resolved at dispatch).
_num_cores: int = 0

# Cache of compiled DataParallel models keyed by (n, inverse, num_cores).
# On CPU this stays empty; on Neuron hardware the first call per key is slow
# (torch_neuronx.trace + DataParallel init), subsequent calls are fast.
_dp_model_cache: dict[tuple, object] = {}


def set_multicore(enabled: bool, num_cores: int = 0) -> None:
    """Enable or disable multi-NeuronCore dispatch.

    Args:
        enabled: If True, `multi_core_fft` routes to the batch-split path.
        num_cores: Number of NeuronCores to target. 0 = use all available
            (resolved via torch_neuronx.get_neuron_device_count() at dispatch
            time, or falls back to min(batch, 2) on CPU).
    """
    global _use_multicore, _num_cores
    _use_multicore = enabled
    _num_cores = num_cores


def get_multicore() -> bool:
    return _use_multicore


def multi_core_fft(x: ComplexTensor, inverse: bool = False) -> ComplexTensor:
    """Dispatch FFT across multiple NeuronCores.

    For batched inputs (batch > 1), runs data-parallel across NeuronCores.
    For single transforms, raises NotImplementedError — stage parallelism is
    not yet implemented (requires inter-core all-reduce infrastructure).

    Falls back to single-core when multicore is disabled or when batch == 1.
    """
    from ..fft_core import fft_core

    if not _use_multicore:
        return fft_core(x, inverse=inverse)

    batch_size = x.real.shape[0] if x.real.dim() > 1 else 1
    if batch_size > 1:
        return _batch_split_fft(x, inverse, _resolve_num_cores(batch_size))

    raise NotImplementedError(
        "Single-transform multi-core FFT not yet implemented. "
        "Stage parallelism requires inter-core all-reduce; use batch dimension "
        "for multi-core speedup."
    )


def _resolve_num_cores(batch_size: int) -> int:
    """Return actual core count: requested, hardware-detected, or batch-capped."""
    if _num_cores > 0:
        return min(_num_cores, batch_size)
    if HAS_TORCH_NEURONX:
        try:
            import torch_neuronx

            hw_cores = torch_neuronx.get_neuron_device_count()
            return min(hw_cores, batch_size)
        except AttributeError:
            pass
    # CPU fallback: treat as 2-core (split in half for testing).
    return min(2, batch_size)


def _batch_split_fft(x: ComplexTensor, inverse: bool, num_cores: int) -> ComplexTensor:
    """Data-parallel FFT: split batch across NeuronCores.

    On Neuron hardware (torch_neuronx available): compiles an FFTModule once per
    (n, inverse, num_cores) key and dispatches via torch_neuronx.DataParallel.
    Subsequent calls on the same shape hit the model cache.

    On CPU (no torch_neuronx): splits the batch into `num_cores` shards and
    processes them sequentially. Output is identical to single-core; this path
    is for correctness testing and as the reference implementation.
    """
    from ..complex import ComplexTensor
    from ..fft_core import fft_core

    real = x.real  # (batch, n)
    imag = x.imag
    n = real.shape[-1]
    batch = real.shape[0]

    actual_cores = min(num_cores, batch)
    if actual_cores <= 1:
        return fft_core(x, inverse=inverse)

    # Split batch into shards.
    real_shards = real.chunk(actual_cores, dim=0)
    imag_shards = imag.chunk(actual_cores, dim=0)

    if HAS_TORCH_NEURONX:
        results = _neuron_dp_dispatch(real_shards, imag_shards, n, inverse, actual_cores)
    else:
        # CPU sequential path: process each shard independently.
        results = [
            fft_core(ComplexTensor(r, i), inverse=inverse)
            for r, i in zip(real_shards, imag_shards, strict=True)
        ]

    out_real = torch.cat([r.real for r in results], dim=0)
    out_imag = torch.cat([r.imag for r in results], dim=0)
    return ComplexTensor(out_real, out_imag)


class _FFTModule(torch.nn.Module):
    """Thin nn.Module wrapper for fft_core, required by torch_neuronx.DataParallel."""

    def __init__(self, n: int, inverse: bool) -> None:
        super().__init__()
        self.n = n
        self.inverse = inverse

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from ..complex import ComplexTensor
        from ..fft_core import fft_core

        y = fft_core(ComplexTensor(real, imag), inverse=self.inverse)
        return y.real, y.imag


def _neuron_dp_dispatch(
    real_shards: tuple[torch.Tensor, ...],
    imag_shards: tuple[torch.Tensor, ...],
    n: int,
    inverse: bool,
    num_cores: int,
) -> list[ComplexTensor]:
    """Dispatch shards via torch_neuronx.DataParallel (Neuron hardware only).

    Compiles and caches a DataParallel model on first call for a given
    (n, inverse, num_cores) key. Subsequent calls skip compilation.
    """
    import torch_neuronx

    from ..complex import ComplexTensor

    cache_key = (n, inverse, num_cores)
    if cache_key not in _dp_model_cache:
        shard_size = real_shards[0].shape[0]
        sample = torch.zeros(shard_size, n)
        module = _FFTModule(n, inverse)
        traced = torch.jit.trace(module, [sample, sample])
        neuron_model = torch_neuronx.trace(traced, [sample, sample])
        dp_model = torch_neuronx.DataParallel(neuron_model)
        _dp_model_cache[cache_key] = dp_model

    dp_model = _dp_model_cache[cache_key]

    results = []
    for r_shard, i_shard in zip(real_shards, imag_shards, strict=True):
        r_out, i_out = dp_model(r_shard, i_shard)
        results.append(ComplexTensor(r_out, i_out))
    return results
