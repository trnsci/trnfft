"""Host cost of one native kernel call: the `trnfft::` custom op against the
cached eager handle (`TRNFFT_NKI_EAGER_HANDLE`, trnblas#92), with aten for
scale, on device-resident inputs.

``issue`` is the host time per call of n back-to-back calls; ``e2e`` the
median of one call plus a synchronize; ms. ``same_as_op`` compares the
handle's result with the op's bit for bit. Inputs are created on the host
and copied to the device (aws-neuron/aws-neuron-sdk#1439).

    python scripts/bench_eager_handle.py
"""

from __future__ import annotations

import statistics
import time

import torch


def _sync():
    import torch_neuronx

    sync = getattr(torch_neuronx, "synchronize", None)
    if sync is None:
        sync = torch_neuronx.neuron.synchronize
    sync()


def _issue_e2e(fn, n=200, e2e_reps=100):
    for _ in range(10):
        fn()
    _sync()
    issue = []
    for _ in range(5):
        t0 = time.perf_counter()
        rs = [fn() for _ in range(n)]
        issue.append((time.perf_counter() - t0) / n * 1e3)
        _sync()
        del rs
    e2e = []
    for _ in range(e2e_reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        e2e.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(issue), statistics.median(e2e)


def _host(out):
    if isinstance(out, tuple | list):
        return tuple(_host(o) for o in out)
    return out.detach().cpu() if isinstance(out, torch.Tensor) else out


def _same(a, b):
    if isinstance(a, tuple):
        return all(_same(x, y) for x, y in zip(a, b, strict=True))
    return torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b


def run(D, cases, n=200):
    """cases: (name, fn, aten) — aten=True runs once (no op/handle toggle)."""
    print(f"{'case':44s} {'path':7s} {'issue':>8s} {'e2e':>8s}  same_as_op", flush=True)
    for name, fn, aten, *kw in cases:
        reps = kw[0] if kw else n
        paths = ("aten",) if aten else ("op", "handle")
        ref = None
        for path in paths:
            D._EAGER_HANDLE_ENABLED = path != "op"
            out = _host(fn())
            same = ""
            if path == "op":
                ref = out
            elif path == "handle":
                same = "yes" if _same(out, ref) else "NO"
            del out
            issue, e2e = _issue_e2e(fn, n=reps, e2e_reps=min(100, 5 * reps))
            print(f"{name:44s} {path:7s} {issue:8.4f} {e2e:8.4f}  {same}", flush=True)
    D._EAGER_HANDLE_ENABLED = True


def main():
    import trnfft
    from trnfft import ComplexTensor
    from trnfft.nki import dispatch as D

    trnfft.set_backend("nki")
    g = torch.Generator().manual_seed(0)

    def dev(*s):
        return torch.randn(*s, generator=g).to("neuron")

    a = [dev(128, 128) for _ in range(4)]
    b, n, stage = 128, 1024, 3
    m = 1 << (stage + 1)
    rows, half = b * (n // m), m >> 1
    x = [dev(b, n), dev(b, n), dev(rows, half), dev(rows, half)]
    X = ComplexTensor(dev(128, 1024), dev(128, 1024))

    def fft():
        y = trnfft.fft(X)
        return y.real, y.imag

    run(
        D,
        [
            ("aten a@b 128x128x128", lambda: a[0] @ a[2], True),
            ("complex_gemm 128x128x128", lambda: D._launch(None, "complex_gemm", *a), False),
            (
                "butterfly_stage 128x1024 stage 3 (n, stage)",
                lambda: D._launch(None, "butterfly_stage", *x, n, stage),
                False,
            ),
            ("trnfft.fft 128x1024 (whole driver)", fft, False, 20),
        ],
    )


if __name__ == "__main__":
    main()
