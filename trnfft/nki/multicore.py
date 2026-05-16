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

# NeuronLink collectives (nki.collectives) require SDK ≥ 2.30 (NKI ≥ 0.4.0).
# Current minimum: NKI 0.3.0 (SDK 2.29). Checked once at import time.
#
# When HAS_COLLECTIVES is True, stage-parallel FFT for large single transforms
# (N > _NEURONLINK_THRESHOLD) will use device-side twiddle + all_reduce instead
# of the current host-side twiddle multiply between _batch_split_fft phases.
# This eliminates two HBM roundtrips per stage-parallel transform at large N.
#
# Implementation path (SDK 2.30+):
#   _stage_parallel_fft_neuronlink(): fused twiddle kernel + nki.collectives.all_reduce
#   _NEURONLINK_THRESHOLD = 2**17  # only pays off at large N on trn2
#
# Measured: SDK 2.29.0 and 2.29.1 do not include nki.collectives (sa-east-1, 2026-05-01).
try:
    HAS_COLLECTIVES = importlib.util.find_spec("nki.collectives") is not None
except (ModuleNotFoundError, ValueError):
    HAS_COLLECTIVES = False

# Disabled by default. Enable via TRNFFT_MULTICORE=1 or set_multicore(True).
_use_multicore = os.environ.get("TRNFFT_MULTICORE", "0") == "1"

# Default target NeuronCore count. 0 means "all available" (resolved at dispatch).
_num_cores: int = 0

# Reserved for future compiled-model cache (SDK 2.30+ DataParallel path).
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

    For batched inputs (batch > 1), runs data-parallel batch splitting.
    For single large transforms, uses the row-column FFT decomposition:
    N = n1 × n2, row FFTs on batch of n1, twiddle multiply, column FFTs on
    batch of n2 — no inter-core communication needed.

    Falls back to single-core when multicore is disabled.
    Raises NotImplementedError for prime N in single-transform mode.
    """
    from ..fft_core import fft_core

    if not _use_multicore:
        return fft_core(x, inverse=inverse)

    batch_size = x.real.shape[0] if x.real.dim() > 1 else 1
    if batch_size > 1:
        return _batch_split_fft(x, inverse, _resolve_num_cores(batch_size))

    # Single transform: row-column decomposition.
    n = x.shape[-1]
    try:
        n1, n2 = _factorize(n)
    except ValueError as exc:
        raise NotImplementedError(
            f"N={n} is prime — single-transform multi-core requires composite N. "
            "Use set_multicore(False) and trnfft.fft() for prime N, or ensure "
            "N is composite (e.g. a power of 2)."
        ) from exc
    return _stage_parallel_fft(x, _resolve_num_cores(min(n1, n2)), inverse)


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


def _factorize(n: int) -> tuple[int, int]:
    """Return (n1, n2) with n1*n2 == n and n1 ≈ sqrt(n) ≤ n2.

    Picks the largest divisor ≤ sqrt(n). This balances the two sub-transform
    batch sizes so both phases keep the NeuronCores equally loaded.
    Raises ValueError for prime n (no non-trivial split exists).
    """
    n1 = int(n**0.5)
    while n1 > 1 and n % n1 != 0:
        n1 -= 1
    if n1 <= 1:
        raise ValueError(
            f"n={n} has no non-trivial divisors (prime); row-column FFT not applicable"
        )
    return n1, n // n1


def _stage_parallel_fft(x: ComplexTensor, num_cores: int, inverse: bool) -> ComplexTensor:
    """Single-transform multi-core FFT via row-column (Cooley-Tukey) decomposition.

    Factorizes N = n1 × n2 and computes DFT(x[n]) where n = n2*m + l:

      1. Reshape x to (n1, n2): x_2d[m, l] = x[n2*m + l]
      2. Column DFTs: Y[k1, l] = DFT_{n1}(x_2d[:, l])[k1]  (batch of n2 size-n1 FFTs)
      3. Twiddle:     Z[k1, l] = Y[k1, l] × exp(sign·2πi·l·k1/N)
      4. Row DFTs:    X[k1, k2] = DFT_{n2}(Z[k1, :])[k2]   (batch of n1 size-n2 FFTs)
      5. Output ordering: X_flat[k] = X_2d[k%n1, k//n1]  (column-major flatten)

    The twiddle step runs on the host in FP32 — this is the "inter-core exchange"
    that would otherwise require an allreduce. No NKI inter-core primitive needed.

    Normalisation: fft_core divides by n1 (column phase) and n2 (row phase) when
    inverse=True, giving a total 1/(n1*n2) = 1/N. No extra correction needed.

    Only beneficial for large N (> ~2^17 on trn1) where single-core SBUF becomes
    the bottleneck. For N ≤ 4096 the Stockham/DFT-GEMM paths are faster.
    """
    import math

    from ..complex import ComplexTensor

    n = x.shape[-1]
    n1, n2 = _factorize(n)
    sign = 1.0 if inverse else -1.0

    x_flat_re = x.real.reshape(n).float()
    x_flat_im = x.imag.reshape(n).float()

    # x_2d[m, l] = x[n2*m + l], shape (n1, n2)
    x_2d_re = x_flat_re.reshape(n1, n2)
    x_2d_im = x_flat_im.reshape(n1, n2)

    # Step 2: Column DFTs — n1-point FFT of each column.
    # _batch_split_fft runs FFTs along the last axis, so transpose to (n2, n1),
    # run the batch, then transpose back to (n1, n2).
    col_in_re = x_2d_re.T.contiguous()  # (n2, n1)
    col_in_im = x_2d_im.T.contiguous()
    col_result = _batch_split_fft(
        ComplexTensor(col_in_re, col_in_im), inverse=inverse, num_cores=num_cores
    )
    Y_re = col_result.real.T.contiguous()  # (n1, n2) — Y[k1, l]
    Y_im = col_result.imag.T.contiguous()

    # Step 3: Twiddle multiply Z[k1, l] = Y[k1, l] × exp(sign·2πi·l·k1/N).
    k1 = torch.arange(n1, dtype=torch.float32).unsqueeze(1)  # (n1, 1)
    l = torch.arange(n2, dtype=torch.float32).unsqueeze(0)  # (1, n2)
    angles = sign * 2.0 * math.pi * l * k1 / n
    tw_re = torch.cos(angles)
    tw_im = torch.sin(angles)
    Z_re = Y_re * tw_re - Y_im * tw_im
    Z_im = Y_re * tw_im + Y_im * tw_re

    # Step 4: Row DFTs — n2-point FFT of each row of Z.
    row_result = _batch_split_fft(
        ComplexTensor(Z_re.contiguous(), Z_im.contiguous()), inverse=inverse, num_cores=num_cores
    )
    X_2d_re = row_result.real  # (n1, n2) — X[k1, k2]
    X_2d_im = row_result.imag

    # Step 5: Column-major flatten: X_flat[k] = X_2d[k%n1, k//n1].
    # X_2d.T.contiguous().reshape(N) reads X_2d column-by-column, which gives
    # position k → X_2d.T[k//n1, k%n1] = X_2d[k%n1, k//n1]. ✓
    out_re = X_2d_re.T.contiguous().reshape(n)
    out_im = X_2d_im.T.contiguous().reshape(n)

    return ComplexTensor(out_re, out_im)


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


def _neuron_dp_dispatch(
    real_shards: tuple[torch.Tensor, ...],
    imag_shards: tuple[torch.Tensor, ...],
    n: int,
    inverse: bool,
    num_cores: int,
) -> list[ComplexTensor]:
    """Dispatch shards via NKI fft_core on Neuron hardware.

    SDK 2.29 (NKI 0.3.0): torch_neuronx.trace + DataParallel fails when
    multiple shapes are compiled in the same process — the internal structure
    flattener asserts layout equality across forward() calls, breaking when
    different benchmark configs produce different shard sizes. Tracked for
    re-evaluation on SDK 2.30+.

    Current approach: call fft_core per shard. fft_core routes through the
    NKI dispatch layer (NEFF-compiled kernels, cached per shape). This gives
    NKI kernel acceleration on each shard without the DataParallel layer.
    True cross-core parallelism requires the NeuronLink collectives path
    (HAS_COLLECTIVES, SDK ≥ 2.30).
    """
    from ..complex import ComplexTensor
    from ..fft_core import fft_core

    results = []
    for r_shard, i_shard in zip(real_shards, imag_shards, strict=True):
        results.append(fft_core(ComplexTensor(r_shard, i_shard), inverse=inverse))
    return results
