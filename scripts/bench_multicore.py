"""Six-step large-N FFT and multi-core splits on one Trn2 chip (trnfft#54, #93).

Prints one ``RESULT`` line per measurement:

    RESULT mode=<m> cores=<k> B=<B> N=<N> ms=<warm median> rel=<rel_l2 vs FP64>

Modes (``--modes``, comma separated):
    cpu       ``torch.fft.fft`` on the host, batched with ``--threads`` intra-op
              threads, and batch-parallel (``--threads`` Python threads with one
              intra-op thread each). Pin with ``taskset``.
    single    ``fft_core`` on one core, default dispatch vs ``TRNFFT_SIX_STEP=1``,
              device-resident input, plus the six-step from host input.
    resident  the batch pre-placed in ``k`` shards on ``neuron:0..k-1``; launches
              issued from one thread (``one``) and from a thread per core
              (``threads``). Compute scaling with no host link in the loop.
    split     ``multicore._batch_split_fft`` (``set_multicore``): host input,
              one thread per core, host in and out.
    proc      one process per core (``NEURON_RT_VISIBLE_CORES=i``): host in/out,
              and resident; the slowest process per rep.
    sixmc     ``multicore._six_step_multicore`` (single transform, both passes
              split across cores through the host), host and device input.
    sixproc   the same six-step split with one process per core and the
              exchange in shared host memory.

Cases are ``B:log2N`` (``--cases``). Needs ``NEURON_RT_VISIBLE_CORES`` covering
the cores used (e.g. ``0-3``) for the multi-core modes. Run ``proc`` and
``sixproc`` in their own invocation: the parent must not open the cores.
"""

# ruff: noqa: B023 -- every lambda here runs before its loop variables change.

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import torch


def _data(B, lg):
    g = torch.Generator().manual_seed(1000 * lg + B)
    return torch.randn(B, 1 << lg, generator=g), torch.randn(B, 1 << lg, generator=g)


def _rel(ref, re, im):
    d = torch.complex(re.double().cpu(), im.double().cpu()) - ref
    return float(d.norm() / ref.norm())


def _time(fn, reps, sync):
    out = fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return out, statistics.median(ts)


def _emit(mode, k, B, lg, t, rel=float("nan")):
    print(f"RESULT mode={mode} cores={k} B={B} N=2^{lg} ms={t * 1e3:.2f} rel={rel:.2e}", flush=True)


def _proc_worker(c, k, B, lg, reps, bar, q):
    import torch_neuronx

    from trnfft.complex import ComplexTensor
    from trnfft.fft_core import fft_core

    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))
    xr, xi = _data(B, lg)
    xr, xi = xr.chunk(k)[c].contiguous(), xi.chunk(k)[c].contiguous()
    dev = torch.device("neuron")
    xd = ComplexTensor(xr.to(dev), xi.to(dev))

    def host():
        o = fft_core(ComplexTensor(xr.to(dev), xi.to(dev)))
        return o.real.cpu(), o.imag.cpu()

    out = {}
    for name, fn in (("proc_host", host), ("proc_resident", lambda: fft_core(xd))):
        fn()
        torch_neuronx.synchronize(dev)
        ts = []
        for _ in range(reps):
            bar.wait()
            t = time.perf_counter()
            fn()
            torch_neuronx.synchronize(dev)
            ts.append(time.perf_counter() - t)
        out[name] = ts
    q.put(out)


def _six_proc_worker(c, k, lg, reps, bufs, bar, q):
    import torch_neuronx  # noqa: F401

    from trnfft.fft_core import _four_step_twiddle_cached, _six_step_pass, _six_step_split
    from trnfft.nki.multicore import _split_range

    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "2")))
    xr, xi, yr, yi, orr, oi = bufs
    n = 1 << lg
    n1, n2 = _six_step_split(n)
    d = torch.device("neuron")
    s, e = _split_range(n1, k)[c]
    s2, e2 = _split_range(n2, k)[c]
    t_re, t_im = _four_step_twiddle_cached(n2, n1, False, torch.float32, d)
    t_re, t_im = t_re[s:e].contiguous(), t_im[s:e].contiguous()
    X = (xr.view(n2, n1), xi.view(n2, n1))
    Y = (yr.view(n1, n2), yi.view(n1, n2))
    O = (orr.view(n1, n2), oi.view(n1, n2))

    def p1():
        a = [X[j][:, s:e].contiguous().to(d).T.contiguous() for j in (0, 1)]
        o_re, o_im = _six_step_pass(a[0], a[1], False, "fast")
        Y[0][s:e] = (o_re * t_re - o_im * t_im).cpu()
        Y[1][s:e] = (o_re * t_im + o_im * t_re).cpu()

    def p2():
        b = [Y[j][:, s2:e2].contiguous().to(d).T.contiguous() for j in (0, 1)]
        o_re, o_im = _six_step_pass(b[0], b[1], False, "fast")
        O[0][:, s2:e2] = o_re.T.contiguous().cpu()
        O[1][:, s2:e2] = o_im.T.contiguous().cpu()

    ts = []
    for _ in range(reps + 1):
        bar.wait()
        t = time.perf_counter()
        p1()
        bar.wait()
        p2()
        bar.wait()
        ts.append(time.perf_counter() - t)
    q.put(ts[1:])


def _spawn(k, target, args_for):
    ctx = mp.get_context("spawn")
    bar = ctx.Barrier(k)
    q = ctx.Queue()
    saved = os.environ.get("NEURON_RT_VISIBLE_CORES")
    procs = []
    for c in range(k):
        # Set before the child starts: importing torch may load torch_neuronx.
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(c)
        p = ctx.Process(target=target, args=(*args_for(c), bar, q))
        p.start()
        procs.append(p)
    if saved is None:
        os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
    else:
        os.environ["NEURON_RT_VISIBLE_CORES"] = saved
    res = [q.get() for _ in procs]
    for p in procs:
        p.join()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="cpu,single")
    ap.add_argument("--cases", default="1:22,1:24,16384:12")
    ap.add_argument("--cores", default="1,2,4")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--default-max-lg",
        type=int,
        default=24,
        help="largest log2 N for the default (Stockham) dispatch in mode single; "
        "its first call compiles for minutes above 2^20",
    )
    args = ap.parse_args()
    modes = args.modes.split(",")
    core_counts = [int(c) for c in args.cores.split(",")]
    torch.set_num_threads(args.threads)
    os.environ["BENCH_THREADS"] = str(args.threads)

    sync = lambda: None  # noqa: E731
    devs = []
    if set(modes) & {"proc", "sixproc"} and set(modes) - {"cpu", "proc", "sixproc"}:
        # A parent that has opened the cores leaves none for the workers.
        ap.error("run proc / sixproc in an invocation without device modes")
    if set(modes) - {"cpu", "proc", "sixproc"}:
        import torch_neuronx

        devs = [torch.device(f"neuron:{i}") for i in range(torch_neuronx.device_count())]

        def sync():
            for d in devs:
                torch_neuronx.synchronize(d)

    from trnfft.complex import ComplexTensor
    from trnfft.fft_core import fft_core
    from trnfft.nki import multicore as mc

    pool = ThreadPoolExecutor(max(4, args.threads))
    for case in args.cases.split(","):
        B, lg = map(int, case.split(":"))
        xr, xi = _data(B, lg)
        ref = torch.fft.fft(torch.complex(xr.double(), xi.double()))
        for mode in modes:
            os.environ.pop("TRNFFT_SIX_STEP", None)
            if mode == "cpu":
                xc = torch.complex(xr, xi)
                out, t = _time(lambda: torch.fft.fft(xc), args.reps, sync)
                _emit("cpu_batched", 0, B, lg, t, _rel(ref, out.real, out.imag))
                if B >= args.threads:
                    torch.set_num_threads(1)
                    parts = xc.chunk(args.threads)
                    _, t = _time(lambda: list(pool.map(torch.fft.fft, parts)), args.reps, sync)
                    torch.set_num_threads(args.threads)
                    _emit("cpu_batchpar", 0, B, lg, t)
            elif mode == "single":
                x = ComplexTensor(xr.to(devs[0]), xi.to(devs[0]))
                for six in ("0", "1") if lg <= args.default_max_lg else ("1",):
                    os.environ["TRNFFT_SIX_STEP"] = six
                    out, t = _time(lambda: fft_core(x), args.reps, sync)
                    _emit(
                        "six_dev" if six == "1" else "default_dev",
                        1,
                        B,
                        lg,
                        t,
                        _rel(ref, out.real, out.imag),
                    )
                    del out
                os.environ["TRNFFT_SIX_STEP"] = "1"
                out, t = _time(lambda: fft_core(ComplexTensor(xr, xi)), args.reps, sync)
                _emit("six_host", 1, B, lg, t, _rel(ref, out.real, out.imag))
                del x, out
            elif mode == "resident":
                os.environ["TRNFFT_SIX_STEP"] = "1"
                for k in core_counts:
                    held = [
                        ComplexTensor(a.to(devs[i]), b.to(devs[i]))
                        for i, (a, b) in enumerate(zip(xr.chunk(k), xi.chunk(k), strict=True))
                    ]
                    outs, t = _time(lambda: [fft_core(h) for h in held], args.reps, sync)
                    got_re = torch.cat([o.real.cpu() for o in outs])
                    got_im = torch.cat([o.imag.cpu() for o in outs])
                    _emit("resident_one", k, B, lg, t, _rel(ref, got_re, got_im))
                    _, t = _time(lambda: list(pool.map(fft_core, held)), args.reps, sync)
                    _emit("resident_threads", k, B, lg, t)
                    del held, outs
            elif mode == "split":
                os.environ["TRNFFT_SIX_STEP"] = "1"
                x = ComplexTensor(xr, xi)
                for k in core_counts:
                    out, t = _time(lambda: mc._batch_split_fft(x, False, k), args.reps, sync)
                    _emit("split_host", k, B, lg, t, _rel(ref, out.real, out.imag))
            elif mode == "proc":
                os.environ["TRNFFT_SIX_STEP"] = "1"
                for k in core_counts:
                    res = _spawn(k, _proc_worker, lambda c: (c, k, B, lg, args.reps))
                    for name in ("proc_host", "proc_resident"):
                        per = [max(r[name][j] for r in res) for j in range(args.reps)]
                        _emit(name, k, B, lg, statistics.median(per))
            elif mode == "sixmc":
                os.environ["TRNFFT_SIX_STEP"] = "1"
                x = ComplexTensor(xr, xi)
                xd = ComplexTensor(xr.to(devs[0]), xi.to(devs[0]))
                for k in core_counts:
                    out, t = _time(lambda: mc._six_step_multicore(x, False, k), args.reps, sync)
                    _emit("sixmc_host", k, B, lg, t, _rel(ref, out.real, out.imag))
                    out, t = _time(lambda: mc._six_step_multicore(xd, False, k), args.reps, sync)
                    _emit("sixmc_dev", k, B, lg, t, _rel(ref, out.real, out.imag))
                del xd
            elif mode == "sixproc":
                assert B == 1, "sixproc runs single transforms"
                os.environ["TRNFFT_SIX_STEP"] = "1"
                n = 1 << lg
                for k in core_counts:
                    bufs = [xr.reshape(n).clone(), xi.reshape(n).clone()]
                    bufs += [torch.empty(n) for _ in range(4)]
                    bufs = [b.share_memory_() for b in bufs]
                    res = _spawn(k, _six_proc_worker, lambda c: (c, k, lg, args.reps, bufs))
                    per = [max(r[j] for r in res) for j in range(args.reps)]
                    _emit(
                        "sixproc_host",
                        k,
                        B,
                        lg,
                        statistics.median(per),
                        _rel(ref, bufs[4].reshape(1, n), bufs[5].reshape(1, n)),
                    )
            else:
                raise ValueError(mode)


if __name__ == "__main__":
    main()
