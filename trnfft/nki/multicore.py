"""Multi-NeuronCore parallelism for large FFT transforms.

**Status: scaffold.** Real implementation requires on-hardware profiling
to determine the right partition strategy and size thresholds.

Partition strategies
--------------------

1. **Data parallelism (batch split)** — Simplest case. When the input has a
   batch dimension > 1, distribute batches across NeuronCores. Each core runs
   the full FFT on its shard. No inter-core communication. Speedup scales
   linearly with core count for batch sizes >= core count.

2. **Stage parallelism (single-transform split)** — Harder. A single large
   FFT is split across cores by distributing butterfly stages. Requires
   inter-core communication after each stage boundary. Only wins when the
   transform is too large to fit in a single core's SBUF.

3. **Hybrid** — For very large batched transforms, combine both.

Size thresholds
---------------

On trn1 (16 NeuronCores, 32 MB SBUF each):
- Transforms up to N=2^16 (64K) fit comfortably in one core
- Stage parallelism only pays off for N > 2^17 or so, and only with
  high-bandwidth HBM access patterns

These thresholds need measurement on real hardware.

Current implementation
----------------------

This module provides the dispatch entry point `multi_core_fft()` which
selects between single-core and multi-core paths. For now, only the
batch-split data-parallel path is scaffolded; single-transform splitting
raises NotImplementedError.
"""

from __future__ import annotations

import os
import torch

from ..complex import ComplexTensor


# Disabled by default. Enable via TRNFFT_MULTICORE=1 or set_multicore(True).
_use_multicore = os.environ.get("TRNFFT_MULTICORE", "0") == "1"


def set_multicore(enabled: bool):
    """Enable or disable multi-NeuronCore dispatch."""
    global _use_multicore
    _use_multicore = enabled


def get_multicore() -> bool:
    return _use_multicore


def multi_core_fft(x: ComplexTensor, inverse: bool = False) -> ComplexTensor:
    """Dispatch FFT across multiple NeuronCores.

    For batch sizes >= num_cores, runs data-parallel. For single large
    transforms exceeding single-core capacity, not yet implemented.
    """
    if not _use_multicore:
        from ..fft_core import fft_core
        return fft_core(x, inverse=inverse)

    batch_size = x.shape[0] if x.real.dim() > 1 else 1
    if batch_size > 1:
        return _batch_split_fft(x, inverse)

    raise NotImplementedError(
        "Single-transform multi-core FFT not yet implemented. "
        "Requires hardware profiling to determine stage-split thresholds."
    )


def _batch_split_fft(x: ComplexTensor, inverse: bool) -> ComplexTensor:
    """Data-parallel FFT: split batch across NeuronCores.

    Scaffold: currently just calls single-core fft_core. On hardware, this
    should use torch_neuronx parallelism primitives to distribute shards
    across NeuronCores.
    """
    from ..fft_core import fft_core
    # TODO: Replace with torch_neuronx.parallel or similar when validated.
    return fft_core(x, inverse=inverse)
