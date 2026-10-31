"""Minimal repro: a kernel input read with ``nl.load_transpose2d`` is slow when
its address changes between launches (trnfft#97).

Standalone: needs only torch-neuronx 2.14+ (native TorchNeuron, Trn2/Trn3)
and nki 0.7. No trnfft import.

Each kernel reads an (M, K) FP32 input tile by tile (128 rows at a time) and
writes it back out. The kernels differ only in how they read:

    plain       ``nl.load`` of contiguous (128, K) row tiles
    t2d         ``nl.load_transpose2d`` of the same tiles, giving (K, 128)
    ap_t        ``nisa.dma_copy`` with an element-strided access pattern that
                gives the same (K, 128) transposed tile with an ordinary DMA
    half_rows   ``nl.load`` of the first K/2 columns of each row (one short
                descriptor per row instead of one per tile)
    pe_t        ``nl.load`` plus ``nisa.nc_transpose`` on the Tensor engine
                (the workaround), giving the same (K, 128) tile

For every kernel and (M, K) it times 8 launches with the same input tensor
and 8 launches alternating between two input tensors of the same shape
(each launch synchronised, median reported), all after warm-up:

    RESULT kernel=<k> M=<M> K=<K> same_ms=<median> alt_ms=<median> ratio=<alt/same>

To see the cause, run it under the system profiler, then summarise:

    neuron-explorer inspect -o <dir> python repro_dma_ring_rebuild.py --profile
    python repro_dma_ring_rebuild.py --summarize <dir> [--kernels <same list>]

``--profile`` runs one shape (32768x64 unless ``--shapes``). Per kernel the
executions are: A, B (the correctness checks), then A x4, then A, B, A, B.
``--summarize`` prints, per execution, the runtime's ``kbl_exec_pre``
duration, the host->device copies inside it (``dmem_buf_copyin``) and the
IO DMA ring allocations inside it, then a per-kernel summary.
"""

import argparse
import json
import os
import statistics
import time

KERNELS = ("plain", "t2d", "ap_t", "half_rows", "pe_t")


def build_ops():
    import nki
    import nki.isa as nisa
    import nki.language as nl
    import torch
    from torch_neuronx import nki_op

    P = 128

    @nki.jit
    def k_plain(x):
        M, K = x.shape
        out = nl.ndarray((M, K), dtype=x.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // P):
            t = nl.load(x[m * P : m * P + P, 0:K])
            nl.store(out[m * P : m * P + P, 0:K], value=t)
        return out

    @nki.jit
    def k_t2d(x):
        M, K = x.shape
        out = nl.ndarray((M // P * K, P), dtype=x.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // P):
            t = nl.load_transpose2d(x[m * P : m * P + P, 0:K])  # (K, P)
            nl.store(out[m * K : m * K + K, 0:P], value=t)
        return out

    @nki.jit
    def k_ap_t(x):
        M, K = x.shape
        out = nl.ndarray((M // P * K, P), dtype=x.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // P):
            t = nl.ndarray((K, P), dtype=x.dtype, buffer=nl.sbuf)
            # Partition j <- column j (stride 1), free i <- row i (stride K).
            nisa.dma_copy(dst=t, src=x.ap(pattern=[[1, K], [K, P]], offset=m * P * K))
            nl.store(out[m * K : m * K + K, 0:P], value=t)
        return out

    @nki.jit
    def k_half_rows(x):
        M, K = x.shape
        H = K // 2
        out = nl.ndarray((M, H), dtype=x.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // P):
            t = nl.load(x[m * P : m * P + P, 0:H])
            nl.store(out[m * P : m * P + P, 0:H], value=t)
        return out

    @nki.jit
    def k_pe_t(x):
        M, K = x.shape
        out = nl.ndarray((M // P * K, P), dtype=x.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // P):
            t = nl.load(x[m * P : m * P + P, 0:K])  # (P, K)
            pt = nl.ndarray((K, P), dtype=x.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=pt, data=t, engine=nisa.engine.tensor)
            st = nl.ndarray((K, P), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=st, src=pt)
            nl.store(out[m * K : m * K + K, 0:P], value=st)
        return out

    def t_shape(x):
        return x.new_empty((x.shape[0] // P * x.shape[1], P))

    @nki_op("ringrepro::plain", mutates_args={}, fake_fn=lambda x: torch.empty_like(x))
    def op_plain(x: torch.Tensor) -> torch.Tensor:
        return k_plain(x)

    @nki_op("ringrepro::t2d", mutates_args={}, fake_fn=t_shape)
    def op_t2d(x: torch.Tensor) -> torch.Tensor:
        return k_t2d(x)

    @nki_op("ringrepro::ap_t", mutates_args={}, fake_fn=t_shape)
    def op_ap_t(x: torch.Tensor) -> torch.Tensor:
        return k_ap_t(x)

    @nki_op(
        "ringrepro::half_rows",
        mutates_args={},
        fake_fn=lambda x: x.new_empty((x.shape[0], x.shape[1] // 2)),
    )
    def op_half_rows(x: torch.Tensor) -> torch.Tensor:
        return k_half_rows(x)

    @nki_op("ringrepro::pe_t", mutates_args={}, fake_fn=t_shape)
    def op_pe_t(x: torch.Tensor) -> torch.Tensor:
        return k_pe_t(x)

    ops = {"plain": op_plain, "t2d": op_t2d, "ap_t": op_ap_t, "half_rows": op_half_rows}
    ops["pe_t"] = op_pe_t
    ref = {
        "plain": lambda x: x,
        "half_rows": lambda x: x[:, : x.shape[1] // 2],
    }
    for k in ("t2d", "ap_t", "pe_t"):
        ref[k] = lambda x: x.reshape(-1, P, x.shape[1]).transpose(1, 2).reshape(-1, P)
    return ops, ref


def run(args):
    import torch
    import torch_neuronx

    dev = torch.device("neuron")
    ops, ref = build_ops()
    kernels = args.kernels.split(",")

    def sync():
        torch_neuronx.synchronize(dev)

    for shape in args.shapes.split(","):
        M, K = map(int, shape.split("x"))
        g = torch.Generator().manual_seed(M + K)
        ha, hb = torch.randn(M, K, generator=g), torch.randn(M, K, generator=g)
        a, b = ha.to(dev), hb.to(dev)
        sync()
        for name in kernels:
            op = ops[name]
            assert torch.equal(op(a).cpu(), ref[name](ha)), name
            assert torch.equal(op(b).cpu(), ref[name](hb)), name

            def timed(seq, op=op):
                ts = []
                for x in seq:
                    t = time.perf_counter()
                    op(x)
                    sync()
                    ts.append((time.perf_counter() - t) * 1e3)
                return statistics.median(ts)

            if args.profile:
                for x in [a] * 4 + [a, b] * 2:
                    op(x)
                    sync()
                print(f"PROFILED kernel={name} M={M} K={K} order=AAAA,ABAB", flush=True)
                continue
            for x in (a, b, a, b):  # warm both addresses
                op(x)
            sync()
            same = min(timed([a] * args.calls), timed([a] * args.calls))
            alt = min(timed([a, b] * (args.calls // 2)), timed([a, b] * (args.calls // 2)))
            print(
                f"RESULT kernel={name} M={M} K={K} same_ms={same:.3f} alt_ms={alt:.3f} "
                f"ratio={alt / same:.2f}",
                flush=True,
            )
        del a, b


def summarize(path, kernels):
    """Per kernel execution: ``kbl_exec_pre``, host->device copy-ins and IO DMA
    ring allocations inside it, from ``neuron-explorer view --output-format json``.

    ``path`` is the JSON file, or the ``inspect`` output dir (then this runs
    ``neuron-explorer view`` itself). Kernels are labelled in order of first
    execution, which is the ``--kernels`` order of the profiled run.
    """
    import subprocess

    if os.path.isdir(path):
        out = os.path.join(path, "system_profile.json")
        if not os.path.exists(out):
            subprocess.run(
                ["neuron-explorer", "view", "-d", path, "--output-format", "json"]
                + ["--output-file", out],
                check=True,
            )
        path = out
    with open(path) as fh:
        d = json.load(fh)
    ev = d["trace_event"]
    pre = sorted((e for e in ev if e["name"] == "kbl_exec_pre"), key=lambda e: e["timestamp"])
    copyin = [e for e in ev if e["name"] == "dmem_buf_copyin"]
    rings = [a for a in d["hbm_allocation"] if a["usage_type"] == "DMA Rings IO"]
    names = {}
    for e in pre:
        names.setdefault(e["model_name"], kernels[len(names)] if len(names) < len(kernels) else "?")
    rows = {}
    for e in pre:
        t0, t1 = e["timestamp"], e["timestamp"] + e["duration"]
        ci = [c for c in copyin if t0 <= c["timestamp"] <= t1]
        rg = [r for r in rings if t0 <= r["start_ts"] <= t1]
        rows.setdefault(names[e["model_name"]], []).append(
            (
                e["duration"] / 1e3,
                len(ci),
                sum(c["size"] for c in ci),
                len(rg),
                sum(r["end_address"] - r["start_address"] for r in rg),
            )
        )
    print("| kernel | exec | kbl_exec_pre us | copy-ins | bytes in | IO ring allocs | ring bytes |")
    print("|---|---|---|---|---|---|---|")
    for k, rs in rows.items():
        for i, (us, nci, bci, nrg, brg) in enumerate(rs):
            print(f"| {k} | {i} | {us:.0f} | {nci} | {bci} | {nrg} | {brg} |")
    print()
    print("| kernel | executions | rebuilds (ring allocs > 0) | median kbl_exec_pre us | max us |")
    print("|---|---|---|---|---|")
    for k, rs in rows.items():
        us = [r[0] for r in rs]
        print(
            f"| {k} | {len(rs)} | {sum(1 for r in rs if r[3])} | "
            f"{statistics.median(us):.0f} | {max(us):.0f} |"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="8192x64,32768x64,16384x128")
    ap.add_argument("--kernels", default=",".join(KERNELS))
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--summarize", metavar="DIR")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize, args.kernels.split(","))
        return
    if args.profile and args.shapes == ap.get_default("shapes"):
        args.shapes = "32768x64"
    run(args)


if __name__ == "__main__":
    main()
