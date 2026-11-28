"""NKI kernel of the double-word FFT (``trnfft.dw``, trnfft#172).

One launch runs the whole transform: every Stockham stage, in SBUF, on
double-word complex data (four fp32 planes: re_hi, re_lo, im_hi, im_lo).

The kernel runs the reference's own butterfly code (``trnfft.dw._butterfly``)
through ``_Ops``, which issues one engine instruction per fp32 operation:
``tensor_tensor`` for add / subtract / multiply and a single-op
``tensor_scalar`` for the Veltkamp constant. Nothing is fused (VectorE has
no FMA; the ScalarE fused multiply-add flushes subnormal products and needs
a per-partition factor, which the twiddles here are not) and TensorE is not
used (its fp32 products are not correctly rounded). So the result equals
``dw.reference_fft`` bit for bit when the fp32 add / subtract / multiply
round to nearest even and nothing is subnormal, as E0 measured on Trn2
VectorE and GpSimd (trnsci/trnsci#169).

Layout. Operands are (M, F) float32 in HBM: each partition row holds
Rb = F / n consecutive transforms (the caller's (B, n) rows reshaped, no
copy). Tiles of P partition rows run in turn: 128-row tiles split between
the programs at LNC=2 (the host pads M to a multiple of 128 per program),
or, for M <= 128, one tile of M / 2 rows per program (``layout``). Each
row's operations do not depend on which program runs it, so the result is
the same bits at any LNC. In SBUF,
stage (R, Ns) reads the planes as (P, Rb, R G, Ns) and writes the other
buffer as (P, Rb, G, R Ns): the next stage's input layout, so no
permutation is needed. Each stage runs in chunks of at most ``CW`` elements
per partition (affine loops over rows and groups, the twiddle chunk loaded
once per k range).

Twiddles: ``tw`` is the (1, T) table of ``dw.twiddle_table(n)``; each stage
DMAs its (3, 4, kc) block to every partition (stride-0 partition pattern),
splits the hi parts once, and passes the planes to the butterfly as
broadcast views over (rows, groups).

SBUF per partition: 2 x 4 planes x F x 4 bytes (F <= 4096: 128 KiB), plus
the twiddle chunk (<= 12 KiB at CW = 256) and the chunk temporaries.
"""

from __future__ import annotations

import os

import torch

from .. import dw as _dw
from .dispatch import HAS_NKI

F_TARGET = 2048  # elements per partition per plane, when n allows (F = n above it)
PMAX = 128

ENGINES = ("auto", "split")
# (add, engines) -> kernel attribute; the op name is the attribute minus "_kernel".
KERNEL_NAMES = {
    ("accurate", "auto"): "dw_fft_accurate_kernel",
    ("sloppy", "auto"): "dw_fft_sloppy_kernel",
    ("accurate", "split"): "dw_fft_accurate_split_kernel",
}
KERNEL_DTYPES = {
    k.removesuffix("_kernel"): frozenset({torch.float32}) for k in KERNEL_NAMES.values()
}


def chunk_width(F: int) -> int:
    """Elements per partition per chunk: 512, or 256 at F = 4096 (SBUF)."""
    return 512 if F <= 2048 else 256


def chunks(Rb: int, G: int, Ns: int, CW: int) -> tuple[int, int, int]:
    """(rc, gc, kc): chunk extents over (rows, groups, k), rc gc kc <= CW."""
    if Ns >= CW:
        return 1, 1, CW
    if G * Ns >= CW:
        return 1, CW // Ns, Ns
    return min(Rb, CW // (G * Ns)), G, Ns


def layout(B: int, n: int, nprog: int = 1) -> tuple[int, int, int]:
    """(Rb, P, NT) for B transforms of length n: Rb rows per partition (a power
    of two, F = Rb n <= max(n, F_TARGET)), P partitions per tile, NT tiles.
    The padded batch is NT P Rb >= B.

    With ``nprog`` programs (LNC=2) the tiles split evenly between them: up
    to 128 partition rows as ``nprog`` tiles of ceil(M / nprog) rows, above it
    as 128-row tiles, NT rounded up to a multiple of ``nprog``. No tile
    straddles a 128-row boundary of the operands: under ``nki.simulate`` at
    LNC=2 a second-program tile of fewer than 128 rows starting below row 128
    lost its rows from 128 on (they came back zero), so that layout is never
    generated."""
    rb_max = max(1, F_TARGET // n)
    rows_pp = -(-B // PMAX)  # rows per partition at full partition use
    Rb = 1
    while Rb < min(rows_pp, rb_max):
        Rb *= 2
    M = -(-B // Rb)  # partition rows needed
    if M <= PMAX:
        if nprog > 1 and M >= nprog:
            return Rb, -(-M // nprog), nprog
        return Rb, M, 1
    NT = -(-M // PMAX)
    if nprog > 1:
        NT += (-NT) % nprog
    return Rb, PMAX, NT


def tiles(M: int, n_prog: int) -> tuple[int, int, int]:
    """(P, NT, programs used) of a launch on M partition rows with ``n_prog``
    programs, as the kernel computes it. ``layout`` produces M for which every
    program gets NT / n_prog whole tiles; for any other M every program runs
    every tile (identical writes), which gives the same bits."""
    if M > PMAX:
        P = PMAX  # the host pads M to a multiple of 128 here
    elif n_prog > 1 and M % n_prog == 0:
        P = M // n_prog  # one tile per program, all below partition row 128
    else:
        P = M
    NT = M // P
    return P, NT, (n_prog if NT % n_prog == 0 else 1)


if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    class _Ops:
        """``dw`` ops object: one instruction per fp32 operation, results in
        new SBUF tiles of ``shape`` (or ``out``)."""

        def __init__(self, shape, split=False, engine=None):
            self.shape = shape
            self.engine = nisa.engine.unknown if engine is None else engine
            self._parts = (self, self)
            if split:
                self._parts = (
                    _Ops(shape, engine=nisa.engine.vector),
                    _Ops(shape, engine=nisa.engine.gpsimd),
                )

        def part(self, c):
            return self._parts[c]

        def _tt(self, a, b, op, out):
            d = nl.ndarray(self.shape, dtype=nl.float32, buffer=nl.sbuf) if out is None else out
            nisa.tensor_tensor(dst=d, data1=a, data2=b, op=op, engine=self.engine)
            return d

        def add(self, a, b, out=None):
            return self._tt(a, b, nl.add, out)

        def sub(self, a, b, out=None):
            return self._tt(a, b, nl.subtract, out)

        def mul(self, a, b, out=None):
            return self._tt(a, b, nl.multiply, out)

        def mulc(self, a, c, out=None):
            d = nl.ndarray(self.shape, dtype=nl.float32, buffer=nl.sbuf) if out is None else out
            nisa.tensor_scalar(dst=d, data=a, op0=nl.multiply, operand0=c, engine=self.engine)
            return d

    def _bcast(t, rc, gc):
        """(P, 1, 1, kc) view broadcast to (P, rc, gc, kc)."""
        if rc > 1:
            t = t.broadcast(1, rc)
        if gc > 1:
            t = t.broadcast(2, gc)
        return t

    def _dw_stage(src, dst, tw, st, P, Rb, CW, mode, split):
        R, Ns, G = st.R, st.Ns, st.G
        sv = [s.reshape((P, Rb, R * G, Ns)) for s in src]
        dv = [d.reshape((P, Rb, G, R * Ns)) for d in dst]
        rc, gc, kc = chunks(Rb, G, Ns, CW)
        o = _Ops((P, rc, gc, kc), split=split)
        for ik in range(Ns // kc):  # static: at most 4 k ranges (Ns = 1024, kc = 256)
            k0 = ik * kc
            w = ws = None
            if Ns > 1:
                t = nl.ndarray((P, R - 1, 4, kc), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=t,
                    src=tw.ap(
                        pattern=[[0, P], [4 * Ns, R - 1], [Ns, 4], [1, kc]], offset=st.off + k0
                    ),
                )
                hs = _dw._split(_Ops((P, R - 1, 2, kc)), t[:, :, 0:2, :])
                w, ws = [], []
                for r in range(R - 1):

                    def v(a, r=r, c=0):
                        return _bcast(a[:, r : r + 1, c : c + 1, :], rc, gc)

                    w.append((v(t, c=0), v(t, c=2), v(t, c=1), v(t, c=3)))
                    ws.append(((v(hs[0], c=0), v(hs[1], c=0)), (v(hs[0], c=1), v(hs[1], c=1))))
            for ir in nl.affine_range(Rb // rc):
                for ig in nl.affine_range(G // gc):
                    rows = slice(ir * rc, (ir + 1) * rc)
                    a = [
                        tuple(
                            s[:, rows, r * G + ig * gc : r * G + (ig + 1) * gc, k0 : k0 + kc]
                            for s in sv
                        )
                        for r in range(R)
                    ]
                    y = [
                        tuple(
                            d[:, rows, ig * gc : (ig + 1) * gc, q * Ns + k0 : q * Ns + k0 + kc]
                            for d in dv
                        )
                        for q in range(R)
                    ]
                    _dw._butterfly(o, a, w, ws, y, mode)

    def _dw_fft(xrh, xrl, xih, xil, tw, mode, split=False):
        M, F = xrh.shape
        n = tw.shape[1] // 4  # dw.table_size(n) = 4 n
        Rb = F // n
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        P, NT, n_prog = tiles(M, n_prog)
        per = NT // n_prog
        stages = _dw.plan(n)
        CW = chunk_width(F)
        outs = [nl.ndarray((M, F), dtype=nl.float32, buffer=nl.shared_hbm) for _ in range(4)]
        ins = (xrh, xrl, xih, xil)
        for j in nl.affine_range(per):
            t0 = (nl.program_id(0) * per + j) * P if n_prog > 1 else j * P
            A = [nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf) for _ in range(4)]
            Bf = [nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf) for _ in range(4)]
            for c in range(4):
                nisa.dma_copy(dst=A[c], src=ins[c][t0 : t0 + P, :])
            cur, nxt = A, Bf
            for st in stages:
                _dw_stage(cur, nxt, tw, st, P, Rb, CW, mode, split)
                cur, nxt = nxt, cur
            for c in range(4):
                nisa.dma_copy(dst=outs[c][t0 : t0 + P, :], src=cur[c])
        return outs[0], outs[1], outs[2], outs[3]

    @nki.jit
    def dw_fft_accurate_kernel(xrh, xrl, xih, xil, tw):
        """Forward double-word FFT, AccurateDWPlusDW additions."""
        return _dw_fft(xrh, xrl, xih, xil, tw, "accurate")

    @nki.jit
    def dw_fft_sloppy_kernel(xrh, xrl, xih, xil, tw):
        """Forward double-word FFT, SloppyDWPlusDW additions."""
        return _dw_fft(xrh, xrl, xih, xil, tw, "sloppy")

    @nki.jit
    def dw_fft_accurate_split_kernel(xrh, xrl, xih, xil, tw):
        """``dw_fft_accurate_kernel`` with the real-part chains on VectorE and
        the imaginary-part chains on GpSimd (both RN fp32 on Trn2, E0): the
        same operations in the same order, so the same bits."""
        return _dw_fft(xrh, xrl, xih, xil, tw, "accurate", split=True)

else:
    dw_fft_accurate_kernel = None  # type: ignore[assignment]
    dw_fft_sloppy_kernel = None  # type: ignore[assignment]
    dw_fft_accurate_split_kernel = None  # type: ignore[assignment]


def lnc() -> int:
    """The LNC a launch will see (NEURON_LOGICAL_NC_CONFIG, default 2 on Trn2)."""
    return int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))


def kernel_name(add: str = "accurate", engines: str = "auto") -> str:
    """Attribute name of the kernel for (add, engines); ValueError if none."""
    name = KERNEL_NAMES.get((add, engines))
    if name is None:
        raise ValueError(f"trnfft.dw: no kernel for add={add!r}, engines={engines!r}")
    return name


def kernel(add: str = "accurate", engines: str = "auto"):
    k = globals().get(kernel_name(add, engines))
    if k is None:
        raise RuntimeError("trnfft.dw: the nki package is not importable")
    return k


def pack(x: _dw.DWComplex, nprog: int = 1):
    """The kernel operands of DWComplex ``x`` (..., n): four (M, F) float32
    planes (zero rows appended to fill the layout) and the layout."""
    n = x.shape[-1]
    planes = [t.reshape(-1, n) for t in x]
    B = planes[0].shape[0]
    Rb, P, NT = layout(B, n, nprog)
    Bp = NT * P * Rb
    if Bp != B:
        planes = [torch.cat([t, t.new_zeros(Bp - B, n)]) for t in planes]
    planes = [t.to(torch.float32).contiguous().reshape(NT * P, Rb * n) for t in planes]
    return planes, B


_DEVICE_TABLES: dict[tuple[int, str], torch.Tensor] = {}


def _table_on(n: int, device) -> torch.Tensor:
    """The twiddle table of length n on ``device``, uploaded once."""
    key = (n, str(device))
    t = _DEVICE_TABLES.get(key)
    if t is None:
        t = _DEVICE_TABLES[key] = _dw.twiddle_table(n).to(device)
    return t


def run(
    x: _dw.DWComplex, add: str = "accurate", simulate: bool = False, engines: str = "auto"
) -> _dw.DWComplex:
    """Forward unnormalised double-word FFT of ``x`` through the kernel:
    under ``nki.simulate`` (host in, host out) or on the device (operands
    moved to the neuron device, result left there)."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    name = kernel_name(add, engines).removesuffix("_kernel")
    nprog = lnc()  # nki.simulate also runs NEURON_LOGICAL_NC_CONFIG programs
    planes, B = pack(x, nprog)
    tw = _dw.twiddle_table(n)
    from . import dispatch as D

    if simulate:
        outs = D._simulate_kernel(kernel(add, engines), *planes, tw, op=name)
    else:
        dev = D._NEURON_DEVICE
        dev_planes = [p if p.device.type == dev else p.to(dev) for p in planes]
        outs = D._launch(
            kernel(add, engines), name, *dev_planes, _table_on(n, dev_planes[0].device)
        )
    res = [o.reshape(-1, n)[:B].reshape(*lead, n) for o in outs]
    return _dw.DWComplex(*res)
