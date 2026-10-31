"""Six-step vs default dispatch crossover, and host-input cost (trnfft#94).

Prints one ``RESULT`` line per measurement:

    RESULT path=<p> input=<i> B=<B> N=2^<lg> first_s=<s> ms=<warm median> rel=<rel_l2 vs FP64>

``--path`` selects the dispatch: ``default`` leaves ``TRNFFT_SIX_STEP`` unset,
``old`` sets it to 0 and ``six`` to 1. ``--inputs`` (comma separated):

    device   input already on ``neuron``; output stays there.
    host     ``fft_core`` on a host tensor; output returns to the host.
    manual   the caller moves the batch with ``.to("neuron")``, runs
             ``fft_core`` and copies the result back with ``.cpu()``.

``first_s`` is the wall time of the first call in this process (trace,
compile and load included), measured on the first input mode listed. Run one
case per process with fresh ``TORCH_NEURONX_*_CACHE_DIR`` directories for an
honest first call; later cases in the same process can reuse kernels.
Pin host threads with ``taskset``; ``--threads`` sets the intra-op threads.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time

import torch


def _data(B, lg):
    g = torch.Generator().manual_seed(1000 * lg + B)
    return torch.randn(B, 1 << lg, generator=g), torch.randn(B, 1 << lg, generator=g)


def _rel(ref, re, im):
    d = torch.complex(re.double().cpu(), im.double().cpu()) - ref
    return float(d.norm() / ref.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=("default", "old", "six"), default="default")
    ap.add_argument("--cases", default="1:20", help="B:log2N, comma separated")
    ap.add_argument("--inputs", default="device,host")
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--precision", default="fast")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.path == "default":
        os.environ.pop("TRNFFT_SIX_STEP", None)
    else:
        os.environ["TRNFFT_SIX_STEP"] = "1" if args.path == "six" else "0"

    import torch_neuronx

    import trnfft
    from trnfft.complex import ComplexTensor
    from trnfft.fft_core import fft_core

    trnfft.set_backend("nki")
    dev = torch.device("neuron")

    def sync():
        torch_neuronx.synchronize(dev)

    for case in args.cases.split(","):
        B, lg = map(int, case.split(":"))
        xr, xi = _data(B, lg)
        x64 = torch.complex(xr.double(), xi.double())
        ref = torch.fft.ifft(x64) if args.inverse else torch.fft.fft(x64)
        del x64
        if B == 1:
            xr, xi = xr[0], xi[0]
            ref = ref[0]
        first = None
        for mode in args.inputs.split(","):
            if mode == "device":
                xd = ComplexTensor(xr.to(dev), xi.to(dev))
                sync()

                def fn(xd=xd):
                    return fft_core(xd, inverse=args.inverse, precision=args.precision)
            elif mode == "host":
                xh = ComplexTensor(xr, xi)

                def fn(xh=xh):
                    return fft_core(xh, inverse=args.inverse, precision=args.precision)
            elif mode == "manual":

                def fn(xr=xr, xi=xi):
                    o = fft_core(
                        ComplexTensor(xr.to(dev), xi.to(dev)),
                        inverse=args.inverse,
                        precision=args.precision,
                    )
                    return ComplexTensor(o.real.cpu(), o.imag.cpu())
            else:
                raise SystemExit(f"unknown input mode {mode}")
            t = time.perf_counter()
            out = fn()
            sync()
            t1 = time.perf_counter() - t
            if first is None:
                first = t1
            rel = _rel(ref, out.real, out.imag)
            want = "neuron" if mode == "device" else "cpu"
            assert out.real.device.type == want, (mode, out.real.device)
            del out
            ts = []
            for _ in range(args.reps):
                t = time.perf_counter()
                o = fn()
                sync()
                ts.append(time.perf_counter() - t)
                del o
            print(
                f"RESULT path={args.path} prec={args.precision} inv={int(args.inverse)} "
                f"input={mode} B={B} N=2^{lg} first_s={first:.1f} "
                f"ms={statistics.median(ts) * 1e3:.2f} rel={rel:.2e}",
                flush=True,
            )
            fn = xd = xh = None
            gc.collect()
        del xr, xi, ref
        gc.collect()


if __name__ == "__main__":
    main()
