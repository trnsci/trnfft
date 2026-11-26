#!/usr/bin/env python
"""Double-word FFT bench for the hardware window (trnfft#172).

Per (N, B), median ms over --reps warm calls and ns per point (B N points):

  dw kernel      device-resident DW planes in and out (one launch; K.run)
  dw split       the same with --engines split (re chains VectorE, im GpSimd)
  dw e2e         complex128 on the host -> split, upload, kernel, download,
                 RN64(hi + lo) -> complex128 on the host (dw.fft_dw)
  host64 tN      torch.fft.fft on complex128 on the host, torch threads = N
                 (2 and all by default: the 2-vCPU and full-host baselines)
  host64 rt      the host fp64 route for device-resident data: download the
                 four planes, combine, host fp64 FFT (all threads), split,
                 upload (what precision="double" costs a device pipeline)
  fp32 dev       trnfft "fast" on device-resident float32 (the fp32 path)

Crossover columns: dw e2e / host64 (all threads) and dw kernel / host64 rt.
A ratio < 1 means the double-word arm wins. Also prints correct bits of the
kernel output against host fp64 (a fast sanity check; the mpmath table is
scripts/dw_accuracy.py) and whether it equals the CPU reference bit for bit.

Usage (Trn2, torch-neuronx 2.14+; run at NEURON_LOGICAL_NC_CONFIG=2 and =1):
  python scripts/bench_dw_fft.py [--sizes 4,16,...,4096] [--points 1048576]
      [--batches 1,128] [--reps 20] [--add accurate] [--threads 2,0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(1 << m) for m in range(2, 13)))
    ap.add_argument("--points", type=int, default=1 << 20, help="B = points / N (plus --batches)")
    ap.add_argument("--batches", default="1,128", help="extra fixed batch sizes")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--add", default="accurate", choices=("accurate", "sloppy"))
    ap.add_argument("--threads", default="2,0", help="host fp64 thread counts (0 = all)")
    ap.add_argument("--no-reference", action="store_true", help="skip the CPU bitwise check")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch_neuronx

    from trnfft import dw
    from trnfft.complex import ComplexTensor
    from trnfft.fft_core import fft_core
    from trnfft.nki import dw_fft as K

    dev = torch.device("neuron")
    sync = torch_neuronx.synchronize
    all_threads = torch.get_num_threads()
    threads = [int(t) or all_threads for t in a.threads.split(",")]

    def median_ms(fn, device=True):
        for _ in range(2):
            fn()
        if device:
            sync()
        ts = []
        for _ in range(a.reps):
            if device:
                sync()
            t = time.perf_counter()
            fn()
            if device:
                sync()
            ts.append(time.perf_counter() - t)
        return sorted(ts)[len(ts) // 2] * 1e3

    def host64_roundtrip(yd):
        h = dw.DWComplex(*(t.to("cpu") for t in yd)).to_complex128()
        z = torch.fft.fft(h)
        return dw.to_dw(z).to(dev)

    print(
        f"LNC={K.lnc()} add={a.add} host threads={all_threads} "
        f"os.cpu_count={os.cpu_count()} reps={a.reps}"
    )
    cols = ["N", "B", "dw kernel", "dw split", "dw e2e"] + [f"host64 t{t}" for t in threads]
    cols += ["host64 rt", "fp32 dev", "e2e/host64", "kernel/rt", "bits", "bitwise", "split == auto"]
    print("| " + " | ".join(cols) + " |   (ms; ns/pt in parentheses)")
    print("|---" * len(cols) + "|")
    rows = []
    for n in (int(v) for v in a.sizes.split(",")):
        batches = sorted({max(1, a.points // n)} | {int(b) for b in a.batches.split(",")})
        for B in batches:
            rng = np.random.default_rng(n * 7 + B)
            z = torch.from_numpy(rng.standard_normal((B, n)) + 1j * rng.standard_normal((B, n)))
            x = dw.to_dw(z)
            xd = x.to(dev)
            pts = B * n
            r = {"N": n, "B": B}
            r["dw kernel"] = median_ms(lambda: K.run(xd, a.add))
            if a.add == "accurate":
                r["dw split"] = median_ms(lambda: K.run(xd, a.add, engines="split"))
                ys = dw.DWComplex(*(t.to("cpu") for t in K.run(xd, a.add, engines="split")))
            else:
                r["dw split"] = float("nan")
            r["dw e2e"] = median_ms(lambda: dw.fft_dw(z, add=a.add), device=False)
            for t in threads:
                torch.set_num_threads(t)
                r[f"host64 t{t}"] = median_ms(lambda: torch.fft.fft(z), device=False)
            torch.set_num_threads(all_threads)
            r["host64 rt"] = median_ms(lambda: host64_roundtrip(xd))
            xr = torch.from_numpy(z.real.numpy().astype(np.float32)).to(dev)
            xi = torch.from_numpy(z.imag.numpy().astype(np.float32)).to(dev)
            ct = ComplexTensor(xr, xi)
            r["fp32 dev"] = median_ms(lambda: fft_core(ct, precision="fast"))
            y = dw.DWComplex(*(t.to("cpu") for t in K.run(xd, a.add)))
            if a.add == "accurate":  # the split kernel must give the same bits
                same = all(
                    torch.equal(p.view(torch.int32), q.view(torch.int32)) for p, q in zip(y, ys)
                )
                r["split == auto"] = same
            ref64 = torch.fft.fft(x.to_complex128()).numpy()
            r["bits"] = dw.correct_bits(y.to_complex128().numpy(), ref64)
            if a.no_reference:
                r["bitwise"] = "-"
            else:
                rf = dw.reference_fft(x, a.add)
                r["bitwise"] = all(
                    torch.equal(p.view(torch.int32), q.view(torch.int32)) for p, q in zip(y, rf)
                )
            r["e2e/host64"] = r["dw e2e"] / r[f"host64 t{all_threads}"]
            best_k = min(r["dw kernel"], r["dw split"]) if a.add == "accurate" else r["dw kernel"]
            r["kernel/rt"] = best_k / r["host64 rt"]
            rows.append(r)
            cells = []
            for c in cols:
                v = r.get(c, "-")
                if c in ("N", "B", "bitwise", "split == auto") or v == "-":
                    cells.append(str(v))
                elif c in ("e2e/host64", "kernel/rt"):
                    cells.append(f"{v:.2f}")
                elif c == "bits":
                    cells.append(f"{v:.1f}")
                else:
                    cells.append(f"{v:.3f} ({v * 1e6 / pts:.2f})")
            print("| " + " | ".join(cells) + " |", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {"lnc": K.lnc(), "add": a.add, "threads": all_threads, "rows": rows}, f, indent=1
            )
    best = [r for r in rows if r["e2e/host64"] < 1]
    print(
        "\ncrossover (dw e2e beats host fp64, all threads): "
        + (", ".join(f"N={r['N']} B={r['B']}" for r in best) if best else "none measured")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
