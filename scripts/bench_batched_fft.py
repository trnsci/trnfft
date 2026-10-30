"""Batched 1-D FFT on the native TorchNeuron path vs host torch.fft.fft (trnfft#82).

Device-resident ``(B, N)`` input, warm median of ``--reps`` timed calls with
``torch_neuronx.synchronize()`` on both sides of each call. The "unfused"
column is the pre-fusion dispatch (single DFT-GEMM up to N=1024, then the
unfused four-step driver), selected in the same process by turning off
``_FOUR_STEP_BATCH_FUSED``; "fused" is the default. The CPU column is
``torch.fft.fft`` on complex64 with ``torch.get_num_threads()`` threads
(pin both sides with ``OMP_NUM_THREADS``).

Usage (on a Trn2 host with torch-neuronx 2.14+):
    OMP_NUM_THREADS=2 NEURON_RT_VISIBLE_CORES=0 python scripts/bench_batched_fft.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="256,1024,4096")
    ap.add_argument("--batches", default="1,16,128,1024")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    import torch_neuronx

    import trnfft
    from trnfft import fft_core as fc
    from trnfft.complex import ComplexTensor

    dev = torch.device("neuron")
    sync = torch_neuronx.synchronize

    def median_ms(fn, device=True):
        for _ in range(2):
            fn()
        if device:
            sync()
        ts = []
        for _ in range(args.reps):
            if device:
                sync()
            t = time.perf_counter()
            fn()
            if device:
                sync()
            ts.append(time.perf_counter() - t)
        return sorted(ts)[len(ts) // 2] * 1e3

    print(f"torch threads: {torch.get_num_threads()}")
    print("| N | B | unfused ms | fused ms | CPU ms | unfused rel | fused rel |")
    print("|---|---|---|---|---|---|---|")
    default = fc._FOUR_STEP_BATCH_FUSED
    try:
        for n in (int(v) for v in args.sizes.split(",")):
            for b in (int(v) for v in args.batches.split(",")):
                g = torch.Generator().manual_seed(7 * n + b)
                xr, xi = torch.randn(b, n, generator=g), torch.randn(b, n, generator=g)
                ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
                xc = torch.complex(xr, xi)
                x = ComplexTensor(xr.to(dev), xi.to(dev))
                row = {"cpu": median_ms(lambda: torch.fft.fft(xc), device=False)}
                for name, fused in (("unfused", False), ("fused", True)):
                    fc._FOUR_STEP_BATCH_FUSED = fused
                    row[name] = median_ms(lambda: trnfft.fft(x))
                    y = trnfft.fft(x)
                    got = y.real.cpu().double().numpy() + 1j * y.imag.cpu().double().numpy()
                    row[name + "_rel"] = np.linalg.norm(got - ref) / np.linalg.norm(ref)
                print(
                    f"| {n} | {b} | {row['unfused']:.3f} | {row['fused']:.3f} | {row['cpu']:.3f} "
                    f"| {row['unfused_rel']:.2e} | {row['fused_rel']:.2e} |",
                    flush=True,
                )
    finally:
        fc._FOUR_STEP_BATCH_FUSED = default
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
