"""Spherical harmonic transform benchmark (trnfft#148, trnsci/trnsci#111).

Times the forward (analysis) and inverse (synthesis) scalar SHT of ``F`` real
fields at band limit ``L`` on the Gauss grid (``nlat = L + 1``,
``nlon = 2L + 2``), with the stage split, and checks the result against an
fp64 reference.

Rows (``--rows``, comma-separated):

* ``cpu``: ``trnfft.sht`` on the CPU (fp32 tables, ``torch.fft`` along
  longitude, the per-m torch GEMM loop), once per ``--threads`` entry.
* ``ducc0`` / ``shtns``: the reference packages on the same box, fp64, at the
  same thread counts (``nthreads``), one field per call.
* ``cpu-pad``: as ``cpu`` with the padded latitude stage (``legendre=
  "torch-padded"``), for checking it on the host.
* ``device-aten-pad`` (``--device``): the fair plain-PyTorch device baseline.
  The latitude stage is ``SHT(legendre="torch-padded")``: the per-m problems
  zero-padded into ``--buckets`` size classes, one ``torch.bmm`` per class, so
  about 2B matmul shapes per (L, F) instead of O(L). Longitude stage:
  ``--aten-fft`` (``trnfft``, the same FFTs as ``device-nki``, or ``torch``
  for ``torch.fft`` on the device).
* ``device-aten`` (``--device --allow-unpadded-aten``): the per-m aten GEMM
  loop (2(L+1) matmul calls per direction). It compiles O(L) programs (over
  19 min cold at L = 511), so it is off unless asked for; ``--count-shapes``
  quotes its shape count without running it.
* ``device-nki`` (``--device``): trnfft FFTs and the NKI latitude kernels.

``--count-shapes`` runs nothing on a device: for each L and F it traces the
latitude stage on ``meta`` tensors under a dispatch recorder and prints, for
the per-m loop and each padded strategy, the distinct matmul shapes, the
distinct non-view (op, input shapes) signatures, and the padding overhead in
flops.

``--threads 2,all`` gives the two CPU baselines of trnsci/trnsci#112 (2
pinned threads, all vCPUs). ``--field-chunk`` runs F in chunks (the device
working set at L = 1023, F = 1024 is ~3 x 8.6 GB; per logical core at LNC=2
there are 24 GiB).

Times are warm medians of ``--reps`` calls (``torch.accelerator.synchronize``
around each on the device); the first call (NKI trace + neuronx-cc compile) is
reported separately as ``cold``. The latitude-stage TFLOP/s counts the flops
of the tiles that are computed (polar tiles skipped).

Usage:
    python scripts/bench_sht.py --lmax 127,511 --fields 1,16,128 --rows cpu,ducc0
    python scripts/bench_sht.py --device --lmax 511,1023 --fields 1,16,128,1024 \\
        --rows device-aten,device-nki,cpu,ducc0,shtns --threads 2,all --field-chunk 256
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch


def _threads(spec: str) -> list[int]:
    out = []
    for t in spec.split(","):
        t = t.strip()
        out.append((os.cpu_count() or 1) if t == "all" else int(t))
    return out


def _sync(device: str):
    if device == "cpu":
        return
    acc = getattr(torch, "accelerator", None)
    if acc is not None and acc.is_available():
        acc.synchronize()
        return
    import torch_neuronx

    getattr(torch_neuronx, "synchronize", None) or torch_neuronx.neuron.synchronize()


def _median(fn, reps: int, device: str) -> tuple[float, float]:
    """(cold seconds, warm median seconds)."""
    _sync(device)
    t = time.perf_counter()
    fn()
    _sync(device)
    cold = time.perf_counter() - t
    ts = []
    for _ in range(reps):
        _sync(device)
        t = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t)
    return cold, float(np.median(ts))


def _random_alm(lmax, nfields, seed=0):
    from trnfft.sht import n_alm

    rng = np.random.default_rng(seed)
    n = n_alm(lmax)
    a = rng.standard_normal((nfields, n)) + 1j * rng.standard_normal((nfields, n))
    a[:, : lmax + 1] = a[:, : lmax + 1].real
    return a


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _ducc(lmax, nlat, nlon, nthreads):
    import ducc0

    def synth(alm):
        return np.stack(
            [
                ducc0.sht.experimental.synthesis_2d(
                    alm=a[None],
                    spin=0,
                    lmax=lmax,
                    geometry="GL",
                    ntheta=nlat,
                    nphi=nlon,
                    nthreads=nthreads,
                )[0]
                for a in alm
            ]
        )

    def anal(f):
        return np.stack(
            [
                ducc0.sht.experimental.analysis_2d(
                    map=x[None], spin=0, lmax=lmax, geometry="GL", nthreads=nthreads
                )[0]
                for x in f
            ]
        )

    return synth, anal


def _shtns(lmax, nlat, nlon, nthreads):
    import shtns

    sh = shtns.sht(lmax, lmax, 1, shtns.sht_orthonormal, nthreads)
    sh.set_grid(nlat, nlon, shtns.sht_gauss | shtns.SHT_PHI_CONTIGUOUS, 1e-10)
    flip = sh.cos_theta[0] < 0

    def synth(alm):
        out = np.stack([sh.synth(np.ascontiguousarray(a)) for a in alm])
        return out[:, ::-1] if flip else out

    def anal(f):
        g = f[:, ::-1] if flip else f
        return np.stack([sh.analys(np.ascontiguousarray(x)) for x in g])

    return synth, anal


class _Chunked:
    """Run an SHT plan over F fields in chunks of ``chunk`` fields."""

    def __init__(self, sht, chunk):
        self.s = sht
        self.chunk = chunk

    def forward(self, f):
        from trnfft import ComplexTensor

        if self.chunk <= 0 or f.shape[0] <= self.chunk:
            return self.s.forward(f)
        parts = [self.s.forward(f[i : i + self.chunk]) for i in range(0, f.shape[0], self.chunk)]
        return ComplexTensor(torch.cat([p.real for p in parts]), torch.cat([p.imag for p in parts]))

    def inverse(self, a):
        from trnfft import ComplexTensor

        F = a.real.shape[0]
        if self.chunk <= 0 or F <= self.chunk:
            return self.s.inverse(a)
        return torch.cat(
            [
                self.s.inverse(
                    ComplexTensor(a.real[i : i + self.chunk], a.imag[i : i + self.chunk])
                )
                for i in range(0, F, self.chunk)
            ]
        )


def _stage_split(s, f, a, reps, device):
    """Warm medians of each stage at the given inputs (one chunk)."""
    x = s.longitude_forward(f)
    y = s.legendre_forward(x)
    packed = s.pack_alm(a)
    g = s.legendre_inverse(packed)
    out = {}
    out["fwd_fft"] = _median(lambda: s._rfft(f), reps, device)[1]
    out["fwd_longitude"] = _median(lambda: s.longitude_forward(f), reps, device)[1]
    out["fwd_legendre"] = _median(lambda: s.legendre_forward(x), reps, device)[1]
    out["fwd_unpack"] = _median(lambda: s.unpack_alm(y), reps, device)[1]
    out["inv_pack"] = _median(lambda: s.pack_alm(a), reps, device)[1]
    out["inv_legendre"] = _median(lambda: s.legendre_inverse(packed), reps, device)[1]
    out["inv_longitude"] = _median(lambda: s.longitude_inverse(g), reps, device)[1]
    return out


def _latitude_signatures(s, nfields):
    """Distinct matmul shapes and non-view (op, input shapes) signatures of one
    forward + one inverse latitude stage, traced on ``meta`` tensors."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class _Rec(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.sigs, self.mm, self.calls = set(), set(), 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            leaves = tree_flatten((args, kwargs))[0]
            shp = tuple(tuple(a.shape) if isinstance(a, torch.Tensor) else repr(a) for a in leaves)
            name = func.overloadpacket.__name__
            if not func.is_view:
                self.sigs.add((name, shp))
            if name in ("mm", "bmm", "addmm", "matmul"):
                self.mm.add(shp)
                self.calls += 1
            return func(*args, **kwargs)

    import copy

    s = copy.copy(s)  # meta copies of the tables; the caller's plan keeps its data
    s.table = s.table.to("meta")
    if s.padded is not None:
        s.padded = copy.copy(s.padded).to("meta")
    x = torch.empty((2 * (s.lmax + 1) * s.H, 2 * nfields), device="meta")
    a = torch.empty((s.n_alm, 2 * nfields), device="meta")
    with _Rec() as r:
        s.legendre_forward(x)
        s.legendre_inverse(a)
    return len(r.mm), len(r.sigs), r.calls


def count_shapes(args) -> int:
    """``--count-shapes``: shape counts and padding overhead, no device."""
    import copy

    from trnfft.sht import SHT, PaddedLegendre, problem_rows

    strategies = [b.strip() for b in args.buckets.split(",")] if "," in args.buckets else []
    strategies = strategies or ["pow2", "8", "10", "12"]
    print(
        "| L | F | latitude stage | buckets | matmul calls (fwd+inv) | distinct matmul shapes | "
        "distinct non-view op signatures | flops executed / per-m loop | / NKI (polar tiles skipped) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for L in (int(v) for v in args.lmax.split(",")):
        s = SHT(L, fft="torch", legendre="torch")
        H = s.H
        dense = int(sum(2 * k * H * 2 for k in problem_rows(L)))  # per field
        nki = s.flops(1)
        table = s.table
        for F in (int(v) for v in args.fields.split(",")):
            n_mm, n_sig, calls = _latitude_signatures(s, F)
            print(
                f"| {L} | {F} | per-m loop (`torch`) | - | {calls} | {n_mm} | {n_sig} | "
                f"1.000 | {dense / nki:.3f} |",
                flush=True,
            )
            for strat in strategies:
                b = strat if strat == "pow2" else int(strat)
                sp = copy.copy(s)
                sp.legendre = "torch-padded"
                sp.padded = PaddedLegendre(table, L, H, b)
                ex = sp.padded.flops(1)
                n_mm, n_sig, calls = _latitude_signatures(sp, F)
                print(
                    f"| {L} | {F} | padded (`torch-padded`) | {strat} ({len(sp.padded.buckets)}) | "
                    f"{calls} | {n_mm} | {n_sig} | {ex / dense:.3f} | {ex / nki:.3f} |",
                    flush=True,
                )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--lmax", default="127,511")
    ap.add_argument("--fields", default="1,16,128")
    ap.add_argument("--rows", default="cpu,ducc0,shtns")
    ap.add_argument("--threads", default="2,all", help="CPU thread counts, e.g. 2,all")
    ap.add_argument("--device", action="store_true", help="enable the device-* rows (Neuron)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--field-chunk", type=int, default=0, help="fields per pass (0 = all)")
    ap.add_argument("--stages", action="store_true", help="also time the stage split")
    ap.add_argument("--check-fields", type=int, default=2, help="fields checked against fp64")
    ap.add_argument("--json", default="", help="append JSON lines here")
    ap.add_argument("--buckets", default="10", help="torch-padded size classes: an int or pow2")
    ap.add_argument("--aten-fft", default="trnfft", choices=("trnfft", "torch"))
    ap.add_argument(
        "--allow-unpadded-aten",
        action="store_true",
        help="allow the device-aten row (per-m loop, O(L) compiles)",
    )
    ap.add_argument(
        "--count-shapes", action="store_true", help="print shape counts (meta, no device) and exit"
    )
    args = ap.parse_args()
    if args.count_shapes:
        return count_shapes(args)
    buckets = args.buckets if args.buckets == "pow2" else int(args.buckets)

    import trnfft
    from trnfft import ComplexTensor
    from trnfft.sht import SHT

    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    if any(r.startswith("device") for r in rows) and not args.device:
        print("device rows need --device", file=sys.stderr)
        return 2
    if "device-aten" in rows and not args.allow_unpadded_aten:
        print(
            "device-aten (the per-m loop) compiles O(L) programs; use device-aten-pad, "
            "or pass --allow-unpadded-aten",
            file=sys.stderr,
        )
        return 2
    threads = _threads(args.threads)
    print(
        f"# host {platform.node()} {platform.machine()} vCPUs={os.cpu_count()} torch={torch.__version__}"
    )
    print(
        "| row | threads | L | F | fwd ms | inv ms | fwd cold s | inv cold s | lat TFLOP/s fwd | "
        "fwd err | inv err |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    jsonf = open(args.json, "a") if args.json else None  # noqa: SIM115

    for L in (int(v) for v in args.lmax.split(",")):
        ref = SHT(L, dtype=torch.float64, fft="torch", legendre="torch")
        nlat, nlon = ref.nlat, ref.nlon
        check_alm = _random_alm(L, args.check_fields, seed=11)
        check_f = ref.inverse(
            ComplexTensor(
                torch.from_numpy(check_alm.real.copy()), torch.from_numpy(check_alm.imag.copy())
            )
        ).numpy()
        del ref
        plans = {}
        for F in (int(v) for v in args.fields.split(",")):
            alm = _random_alm(L, F, seed=1)
            for row in rows:
                for nt in threads if row in ("cpu", "cpu-pad", "ducc0", "shtns") else [None]:
                    rec = {"row": row, "threads": nt, "lmax": L, "fields": F}
                    if row in ("ducc0", "shtns"):
                        try:
                            synth, anal = (_ducc if row == "ducc0" else _shtns)(L, nlat, nlon, nt)
                        except ImportError:
                            print(f"| {row} | - | {L} | {F} | not installed |")
                            continue
                        fmap = synth(alm[:1])
                        fmap = np.repeat(fmap, F, axis=0) if F > 1 else fmap
                        c_i, t_i = _median(lambda: synth(alm), args.reps, "cpu")
                        c_f, t_f = _median(lambda: anal(fmap), args.reps, "cpu")
                        e_f = _rel(anal(check_f), check_alm)
                        e_i = _rel(synth(check_alm), check_f)
                        flops_note = ""
                    else:
                        device = "cpu" if row.startswith("cpu") else "neuron"
                        if device == "cpu":
                            torch.set_num_threads(nt)
                            trnfft.set_backend("pytorch")
                            leg = "torch-padded" if row == "cpu-pad" else "torch"
                            key = ("cpu", leg)
                            if key not in plans:
                                plans[key] = SHT(L, fft="torch", legendre=leg, buckets=buckets)
                        else:
                            trnfft.set_backend("nki")
                            leg = {
                                "device-nki": "nki",
                                "device-aten": "torch",
                                "device-aten-pad": "torch-padded",
                            }[row]
                            fft = args.aten_fft if leg == "torch-padded" else "trnfft"
                            key = ("dev", leg, fft)
                            if key not in plans:
                                plans[key] = SHT(
                                    L, legendre=leg, fft=fft, buckets=buckets, device="neuron"
                                )
                        s = plans[key]
                        run = _Chunked(s, args.field_chunk)
                        a_ct = ComplexTensor(
                            torch.from_numpy(alm.real.copy()).float().to(device),
                            torch.from_numpy(alm.imag.copy()).float().to(device),
                        )
                        f_t = run.inverse(a_ct)
                        c_f, t_f = _median(lambda: run.forward(f_t), args.reps, device)
                        c_i, t_i = _median(lambda: run.inverse(a_ct), args.reps, device)
                        cf = torch.from_numpy(check_f).float().to(device)
                        ca = s.forward(cf)
                        e_f = _rel(
                            ca.real.cpu().double().numpy() + 1j * ca.imag.cpu().double().numpy(),
                            check_alm,
                        )
                        ci = s.inverse(
                            ComplexTensor(
                                torch.from_numpy(check_alm.real.copy()).float().to(device),
                                torch.from_numpy(check_alm.imag.copy()).float().to(device),
                            )
                        )
                        e_i = _rel(ci.cpu().double().numpy(), check_f)
                        if args.stages:
                            n = F if args.field_chunk <= 0 else min(F, args.field_chunk)
                            rec["stages_ms"] = {
                                k: v * 1e3
                                for k, v in _stage_split(
                                    s,
                                    f_t[:n],
                                    ComplexTensor(a_ct.real[:n], a_ct.imag[:n]),
                                    args.reps,
                                    device,
                                ).items()
                            }
                            rec["stage_fields"] = n
                        lat = s.flops(F) / 1e12
                        rec["latitude_tflop"] = lat
                        rec["tiles"] = s.tile_counts
                        if s.padded is not None:
                            rec["executed_tflop"] = s.padded.flops(F) / 1e12
                            rec["buckets"] = len(s.padded.buckets)
                            rec["fft"] = s.fft
                        flops_note = ""
                        if "stages_ms" in rec:
                            sf = rec["stages_ms"]["fwd_legendre"] / 1e3 * (F / rec["stage_fields"])
                            flops_note = f"{lat / sf:.2f}"
                        del f_t, a_ct
                    rec.update(
                        fwd_ms=t_f * 1e3,
                        inv_ms=t_i * 1e3,
                        fwd_cold_s=c_f,
                        inv_cold_s=c_i,
                        fwd_err=e_f,
                        inv_err=e_i,
                    )
                    print(
                        f"| {row} | {nt if nt else '-'} | {L} | {F} | {t_f * 1e3:.2f} | {t_i * 1e3:.2f} | "
                        f"{c_f:.1f} | {c_i:.1f} | {flops_note} | {e_f:.2e} | {e_i:.2e} |",
                        flush=True,
                    )
                    if "stages_ms" in rec:
                        st = ", ".join(f"{k}={v:.2f}" for k, v in rec["stages_ms"].items())
                        print(f"|   stages ({rec['stage_fields']} fields, ms): {st} |", flush=True)
                    if jsonf:
                        jsonf.write(json.dumps(rec) + "\n")
                        jsonf.flush()
    trnfft.set_backend("auto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
