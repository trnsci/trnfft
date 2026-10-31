"""Multi-NeuronCore parallelism for FFT transforms.

Partition strategies
--------------------

1. **Data parallelism (batch split)**: for a host batch with more than one
   row, shard ``i`` runs on logical NeuronCore ``neuron:i`` with its own host
   thread (:func:`_batch_split_fft`). There is no inter-core communication.
   A batch already on a NeuronCore is not split.

2. **Single-transform split**: a row-column (four-step) decomposition with
   each phase run as a batch split (:func:`_stage_parallel_fft`), and a
   six-step variant with both passes split across cores and the exchange
   staged through the host (:func:`_six_step_multicore`, benchmark only).

Measured on Trn2 (one chip, 4 logical cores, trnfft#54 / #93), the host link
sets the time, not the cores:

- Host-to-device copies run at about 12 GB/s whether 1 or 4 cores copy.
- The threaded batch split gains at most 1.5x over one core.
- The six-step split is far slower than the single-core six-step
  (``TRNFFT_SIX_STEP=1``): 169 against 9.15 ms at N = 2^24.
- Shards a caller keeps on ``neuron:0..3`` and transforms in place do scale:
  2.2x on 4 cores at B = 16384, N = 4096.

On CPU (no torch_neuronx) each shard is processed sequentially. That path is
architecturally correct but gives no hardware parallelism.
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
            (resolved via torch_neuronx.device_count() at dispatch
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

            count = getattr(torch_neuronx, "device_count", None)
            if count is None:
                count = torch_neuronx.get_neuron_device_count
            return min(count(), batch_size)
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
    k1 = torch.arange(n1, dtype=torch.float32, device=Y_re.device).unsqueeze(1)  # (n1, 1)
    l = torch.arange(n2, dtype=torch.float32, device=Y_re.device).unsqueeze(0)  # (1, n2)
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

    On the native TorchNeuron path a host-resident batch is split into
    ``num_cores`` shards, shard ``i`` on ``neuron:i`` with its own host thread
    (:func:`_neuron_dp_dispatch`). A batch already on a NeuronCore is not
    split: the other cores could only be reached through the host, and on
    Trn2 that staging cost 15-20x the single-core transform (trnfft#54). To
    use several cores, place the shards on ``neuron:0..k-1`` yourself and call
    ``trnfft.fft`` on each; launches stay on each shard's core.

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
    if actual_cores <= 1 or real.device.type == "neuron":
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


_POOL = None
# (n, inverse, shard rows, devices) of the splits that have run once.
_WARM: set = set()


def _pool(workers: int):
    global _POOL
    if _POOL is None or _POOL._max_workers < workers:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trnfft-core")
    return _POOL


def _core_devices(src: torch.device, num_cores: int) -> list[torch.device]:
    """``num_cores`` logical NeuronCores, starting at ``src``'s core when it is one."""
    import torch_neuronx

    ndev = torch_neuronx.device_count()
    k = max(1, min(num_cores, ndev))
    first = (src.index or 0) if src.type == "neuron" else 0
    return [torch.device("neuron", (first + i) % ndev) for i in range(k)]


def _neuron_dp_dispatch(
    real_shards: tuple[torch.Tensor, ...],
    imag_shards: tuple[torch.Tensor, ...],
    n: int,
    inverse: bool,
    num_cores: int,
) -> list[ComplexTensor]:
    """Run host shard ``i`` on logical NeuronCore ``i`` (``neuron:i``), one host thread each.

    Native TorchNeuron path (trnfft#54). The shards are independent rows, so
    the result equals the single-core one; each thread copies its shard in,
    runs ``fft_core`` there and copies the result back to the host. The first
    call for a shard layout runs the shards one after another, because NKI
    tracing is not thread-safe (trnsolver#54); later calls run them
    concurrently. Off the native path (XLA, simulator) the shards run in turn
    through ``fft_core``.

    Trn2 measurements (trnfft#54): the host link, not the cores, sets the
    time. H2D runs at about 12 GB/s whether 1 or 4 cores copy, so the split
    gains at most 1.6x over one core from host input, and no case beat
    ``torch.fft`` on the same 2 host CPUs.
    """
    from ..complex import ComplexTensor
    from ..fft_core import fft_core
    from .dispatch import _is_native, _use_simulator

    if _use_simulator() or not _is_native():
        return [
            fft_core(ComplexTensor(r, i), inverse=inverse)
            for r, i in zip(real_shards, imag_shards, strict=True)
        ]

    src = real_shards[0].device
    devices = _core_devices(src, num_cores)
    k = min(len(devices), len(real_shards))

    def run(i: int) -> ComplexTensor:
        d = devices[i]
        out = fft_core(ComplexTensor(real_shards[i].to(d), imag_shards[i].to(d)), inverse=inverse)
        return ComplexTensor(out.real.to(src), out.imag.to(src))

    key = (n, inverse, tuple(r.shape[0] for r in real_shards[:k]), tuple(map(str, devices[:k])))
    if k == 1 or key not in _WARM:
        results = [run(i) for i in range(k)]
        _WARM.add(key)
    else:
        results = list(_pool(k).map(run, range(k)))
    return results


def _split_range(total: int, parts: int) -> list[tuple[int, int]]:
    """``[0, total)`` in at most ``parts`` near-equal ``[start, stop)`` pieces."""
    parts = max(1, min(parts, total))
    per = -(-total // parts)
    return [(s, min(s + per, total)) for s in range(0, total, per)]


def _six_step_multicore(
    x: ComplexTensor, inverse: bool, num_cores: int, devices: list[torch.device] | None = None
) -> ComplexTensor:
    """Six-step FFT (``fft_core._fft_via_six_step``) with both row passes split
    across logical NeuronCores, the exchange staged through the host (trnfft#54).

    With ``N = N1 * N2`` and the (B, N2, N1) view of x:

    1. Core c takes columns ``j1 in [s, e)``: the host copies the
       ``(B, N2, e-s)`` block (contiguous runs of e-s), the core transposes
       it, runs ``B*(e-s)`` row FFTs of length N2 and the twiddle
       ``W_N^(j1 k2)`` for its rows, and returns ``(B, e-s, N2)``, which the
       host writes into ``Y[:, s:e, :]``.
    2. Core c takes ``k2 in [s, e)``: the host copies ``Y[:, :, s:e]``, the
       core transposes, runs ``B*(e-s)`` row FFTs of length N1, transposes
       back to ``(B, N1, e-s)``, and the host writes ``X[:, :, s:e]``.

    All transposes run on the device; the host only does block copies. The
    full array crosses the host link four times (two H2D, two D2H), which is
    what this path costs over the single-core six-step (see CHANGELOG).
    """
    from ..complex import ComplexTensor
    from ..fft_core import _four_step_twiddle_cached, _six_step_pass, _six_step_split

    n = x.shape[-1]
    n1, n2 = _six_step_split(n)
    orig_shape = x.real.shape
    src = x.real.device
    re = x.real.reshape(-1, n).cpu()
    im = x.imag.reshape(-1, n).cpu()
    B = re.shape[0]
    if devices is None:
        devices = _core_devices(src, num_cores)
    prec = "fast"

    cols = _split_range(n1, len(devices))
    xr3, xi3 = re.reshape(B, n2, n1), im.reshape(B, n2, n1)
    blocks = [(xr3[:, :, s:e].contiguous(), xi3[:, :, s:e].contiguous()) for s, e in cols]
    y_re = torch.empty(B, n1, n2, dtype=re.dtype)
    y_im = torch.empty(B, n1, n2, dtype=re.dtype)

    def pass1(c: int) -> None:
        s, e = cols[c]
        d = devices[c]
        w = e - s
        a_re = blocks[c][0].to(d).transpose(1, 2).contiguous().reshape(B * w, n2)
        a_im = blocks[c][1].to(d).transpose(1, 2).contiguous().reshape(B * w, n2)
        o_re, o_im = _six_step_pass(a_re, a_im, inverse, prec)
        t_re, t_im = _four_step_twiddle_cached(n2, n1, inverse, re.dtype, d)
        t_re, t_im = t_re[s:e], t_im[s:e]
        o_re, o_im = o_re.reshape(B, w, n2), o_im.reshape(B, w, n2)
        z_re = o_re * t_re - o_im * t_im
        z_im = o_re * t_im + o_im * t_re
        y_re[:, s:e, :] = z_re.cpu()
        y_im[:, s:e, :] = z_im.cpu()

    rows = _split_range(n2, len(devices))
    out_re = torch.empty(B, n1, n2, dtype=re.dtype)
    out_im = torch.empty(B, n1, n2, dtype=re.dtype)

    def pass2(c: int) -> None:
        s, e = rows[c]
        d = devices[c]
        w = e - s
        b_re = y_re[:, :, s:e].contiguous().to(d).transpose(1, 2).contiguous().reshape(B * w, n1)
        b_im = y_im[:, :, s:e].contiguous().to(d).transpose(1, 2).contiguous().reshape(B * w, n1)
        o_re, o_im = _six_step_pass(b_re, b_im, inverse, prec)
        out_re[:, :, s:e] = o_re.reshape(B, w, n1).transpose(1, 2).contiguous().cpu()
        out_im[:, :, s:e] = o_im.reshape(B, w, n1).transpose(1, 2).contiguous().cpu()

    key = ("six", n, inverse, B, tuple(map(str, devices)))
    k = len(devices)
    for fn, parts in ((pass1, len(cols)), (pass2, len(rows))):
        if k == 1 or key not in _WARM:
            for c in range(parts):
                fn(c)
        else:
            list(_pool(parts).map(fn, range(parts)))
    _WARM.add(key)
    return ComplexTensor(out_re.reshape(orig_shape).to(src), out_im.reshape(orig_shape).to(src))
