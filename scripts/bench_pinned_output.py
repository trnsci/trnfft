"""Host input, host output: pageable vs caching-host-allocator results (trnfft#97).

For each ``B:log2N`` case, ``fft_core`` runs on a host batch with
``TRNFFT_PINNED_OUTPUT=0`` (fresh pageable result, the old behaviour) and
``=1`` (result from ``torch.empty(..., pin_memory=True)``), alternating
0,1,1,0,... so drift cancels. Prints one line per case:

    RESULT B=<B> N=2^<lg> inv=<0|1> pageable_ms=<median> pinned_ms=<median>
           device_ms=<device-resident median> pinned=<is_pinned> equal=<bitwise>

``equal`` compares the two results bit for bit (``torch.equal`` on the int32
views of both planes). Pin host threads with ``taskset``.
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
    ap.add_argument("--cases", default="16384:12,16384:10,4096:12,1024:12,256:13,1024:13")
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--reps", type=int, default=6, help="per mode, even")
    ap.add_argument("--threads", type=int, default=2)
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

    for case in args.cases.split(","):
        B, lg = map(int, case.split(":"))
        g = torch.Generator().manual_seed(1000 * lg + B)
        xr, xi = torch.randn(B, 1 << lg, generator=g), torch.randn(B, 1 << lg, generator=g)
        if B == 1:
            xr, xi = xr[0], xi[0]
        xh = ComplexTensor(xr, xi)

        def run(mode, xh=xh):
            os.environ["TRNFFT_PINNED_OUTPUT"] = mode
            t = time.perf_counter()
            o = fft_core(xh, inverse=args.inverse)
            sync()
            return time.perf_counter() - t, o

        _, off = run("0")  # warm: compile, load
        _, on = run("1")
        equal = torch.equal(bits(off.real), bits(on.real)) and torch.equal(
            bits(off.imag), bits(on.imag)
        )
        pinned = on.real.is_pinned() and on.imag.is_pinned()
        assert off.real.device.type == "cpu" and not off.real.is_pinned()
        del off, on
        ts = {"0": [], "1": []}
        order = ["0", "1", "1", "0"] * (args.reps // 2)
        for mode in order[: 2 * args.reps]:
            dt, o = run(mode)
            ts[mode].append(dt)
            del o
        os.environ.pop("TRNFFT_PINNED_OUTPUT", None)

        xd = ComplexTensor(xr.to(dev), xi.to(dev))
        fft_core(xd, inverse=args.inverse)
        sync()
        dts = []
        for _ in range(args.reps):
            t = time.perf_counter()
            o = fft_core(xd, inverse=args.inverse)
            sync()
            dts.append(time.perf_counter() - t)
            del o
        del xd
        print(
            f"RESULT B={B} N=2^{lg} inv={int(args.inverse)} "
            f"pageable_ms={statistics.median(ts['0']) * 1e3:.2f} "
            f"pinned_ms={statistics.median(ts['1']) * 1e3:.2f} "
            f"device_ms={statistics.median(dts) * 1e3:.2f} "
            f"pinned={int(pinned)} equal={int(equal)}",
            flush=True,
        )
        del xh, xr, xi
        gc.collect()


if __name__ == "__main__":
    main()
