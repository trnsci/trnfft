"""Small-N FFT with a fresh input tensor per call vs the same tensor (trnfft#97).

On Trn2 (SDK 2.33, native TorchNeuron) a kernel that reads its input with
``nl.load_transpose2d`` makes the runtime rebuild its IO DMA rings whenever
the input's address changes. This times ``fft_core`` (the default N < 512
path is the single DFT-GEMM) for each ``B:N`` case in four input modes:

    same   one device tensor, every call
    alt    two device tensors, alternating
    fresh  eight device tensors in turn, so every call sees a new address
    host   a host tensor; the input is copied to the device on each call
           and the result returned to the host

under ``TRNFFT_PE_TRANSPOSE=0`` (``load_transpose2d`` kernels) and ``=1``
(plain load + Tensor-engine transpose), in rounds 0,1,1,0 in one process.
Each call is synchronised and timed on its own; the figure is the median
call. Prints one line per case and mode:

    RESULT B=<B> N=<N> mode=<mode> t2d_ms=<median> pe_ms=<median> equal=<bitwise>

``equal`` is ``torch.equal`` on the int32 views of both planes (t2d vs pe,
same input). Pin host threads with ``taskset``.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1:64,1024:64,8192:64")
    ap.add_argument("--modes", default="same,alt,fresh,host")
    ap.add_argument("--calls", type=int, default=8, help="timed calls per mode and round")
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--knobs", default="0,1,1,0", help="TRNFFT_PE_TRANSPOSE per round")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import torch_neuronx

    import trnfft
    from trnfft.complex import ComplexTensor
    from trnfft.fft_core import fft_core

    trnfft.set_backend("nki")
    dev = torch.device("neuron")

    def sync():
        torch_neuronx.synchronize(dev)

    def bits(t):
        return t.contiguous().view(torch.int32)

    pool = 8
    for case in args.cases.split(","):
        B, N = map(int, case.split(":"))
        g = torch.Generator().manual_seed(7919 * N + B)
        host = [
            (torch.randn(B, N, generator=g), torch.randn(B, N, generator=g)) for _ in range(pool)
        ]
        if B == 1:
            host = [(r[0], i[0]) for r, i in host]
        devs = [ComplexTensor(r.to(dev), i.to(dev)) for r, i in host]
        hosts = [ComplexTensor(r, i) for r, i in host]
        sync()

        def inputs(mode, k, devs=devs, hosts=hosts):
            if mode == "same":
                return devs[0]
            if mode == "alt":
                return devs[k % 2]
            if mode == "fresh":
                return devs[k % pool]
            return hosts[k % pool]

        for mode in args.modes.split(","):
            times = {"0": [], "1": []}
            outs = {}
            for knob in args.knobs.split(","):
                os.environ["TRNFFT_PE_TRANSPOSE"] = knob
                # Warm: compile, and put the mode's addresses through once.
                for k in range(pool if mode in ("fresh", "host") else 2):
                    y = fft_core(inputs(mode, k), inverse=args.inverse)
                sync()
                for k in range(args.calls):
                    x = inputs(mode, k)
                    t = time.perf_counter()
                    y = fft_core(x, inverse=args.inverse)
                    sync()
                    times[knob].append((time.perf_counter() - t) * 1e3)
                outs[knob] = fft_core(inputs(mode, 0), inverse=args.inverse)
                sync()
                del y
            equal = None
            if "0" in outs and "1" in outs:
                a, b = outs["0"], outs["1"]
                equal = torch.equal(bits(a.real.cpu()), bits(b.real.cpu())) and torch.equal(
                    bits(a.imag.cpu()), bits(b.imag.cpu())
                )

            def med(k, times=times):
                return f"{statistics.median(times[k]):.3f}" if times[k] else "-"

            print(
                f"RESULT B={B} N={N} mode={mode} t2d_ms={med('0')} pe_ms={med('1')} equal={equal}",
                flush=True,
            )
        del devs, hosts, host
        gc.collect()


if __name__ == "__main__":
    main()
