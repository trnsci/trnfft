"""
Double-word complex FFT: fp64-class accuracy from fp32 arithmetic (trnfft#172).

Trainium computes float64 at fp32 (#136), so ``precision="double"`` runs on
the host. This module is the first device arm of #172: every real number is
a double-word (float-float) value, the unevaluated sum ``hi + lo`` of two
fp32 numbers (about 48 significant bits, fp32's exponent range), and every
butterfly is built from error-free transformations in fp32 arithmetic, which
the Trn2 Vector and GpSimd engines perform with IEEE round-to-nearest-even
(the E0 probe, trnsci/trnsci#169).

Building blocks (all FMA-free, one fp32 operation per step):

* Knuth's TwoSum and Dekker's FastTwoSum; Dekker's TwoProd with the
  Veltkamp split (``SPLITTER = 2^12 + 1``).
* Double-word addition: AccurateDWPlusDW (default) or SloppyDWPlusDW, and
  DWTimesDW (FMA-free), from Joldes, Muller and Popescu, "Tight and rigorous
  error bounds for basic building blocks of double-word arithmetic", ACM
  TOMS 44(2), 2017.
* Complex double-word multiplication as a sum of two double-word products
  per component (Stloukal and Jezequel, arXiv:2611.08371, who analyse
  complex double-word multiplication and find that sloppy addition suffices
  for their workload): ``re = ar wr - ai wi``, ``im = ar wi + ai wr``, each
  product a normalised DWTimesDW, each sum a double-word addition of the
  selected kind (``add="accurate"`` or ``"sloppy"``).

Algorithm: a Stockham autosort FFT of power-of-two length n = 2^m,
2 <= n <= 4096, decimation in time, radix 4 (with one leading radix-2 stage
when m is odd). Stage (R, Ns) reads the data as (R G, Ns) and writes it as
(G, R Ns), G = n / (R Ns):

    a_r[g, k] = x[(r G + g) Ns + k]                      r = 0 .. R-1
    b_r       = a_r * w^(r k),  w = exp(-2 pi i / (R Ns))  (skipped when Ns = 1)
    y[g, q Ns + k] = sum_r b_r exp(-2 pi i q r / R)       (adds, subtracts, *(-i))

The output of a stage is already the input layout of the next one, so there
is no permutation. The inverse transform is the forward one on swapped
(re, im) parts, ``ifft(x) n = swap(fft(swap(x)))``, which is exact; the 1/n
is a power of two.

Twiddles: computed on the host in float64 with an octant reduction (exact
zeros, ones and symmetries), then split exactly into ``hi = RN32(w)``,
``lo = RN32(w - hi)``. The Veltkamp split of the twiddles' hi parts is done
once per twiddle tile, with the same fp32 operations as the data's.

Deterministic order: the operation sequence of every output element depends
only on n and ``add``. The NKI kernel (``trnfft/nki/dw_fft.py``) runs the
same Python code as the reference below (``_stage``, through an ops object
that issues one engine instruction per fp32 operation), so its result
equals ``reference_fft`` bit for bit when the engines' fp32 add, subtract
and multiply round to nearest even and nothing is subnormal, at any batch,
tiling and LNC.

Range: the Dekker TwoProd is exact for |a|, |b| <= 2^115 and products above
the fp32 subnormal range (|a b| >= 2^-102 for full precision). With FFT
growth bounded by n, inputs should satisfy max|x| <= 2^100; tiny values
below ~2^-102 lose relative precision gradually (zero is exact). Inputs past
the fp32 range raise ``OverflowError``.

No autograd: the double-word FFT is a forward-only accuracy mode.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

F32 = torch.float32
F64 = torch.float64

SPLITTER = 4097.0  # 2^12 + 1: Veltkamp split of a 24-bit significand into 12 + 12
MIN_N = 2
MAX_N = 4096
ADD_MODES = ("accurate", "sloppy")
U_DW = 2.0**-48  # double-word representation error bound (relative)
FP32_MAX = float(np.finfo(np.float32).max)


class DWComplex(NamedTuple):
    """A double-word complex tensor: value = (re_hi + re_lo) + i (im_hi + im_lo),
    four float32 tensors of one shape on one device."""

    re_hi: torch.Tensor
    re_lo: torch.Tensor
    im_hi: torch.Tensor
    im_lo: torch.Tensor

    @property
    def shape(self):
        return self.re_hi.shape

    @property
    def device(self):
        return self.re_hi.device

    def to(self, device) -> DWComplex:
        return DWComplex(*(t.to(device) for t in self))

    def swap(self) -> DWComplex:
        """(re, im) -> (im, re): the exact conjugation trick for the inverse."""
        return DWComplex(self.im_hi, self.im_lo, self.re_hi, self.re_lo)

    def to_complex128(self) -> torch.Tensor:
        """complex128 on the host: RN64(hi + lo) per part."""
        rh, rl, ih, il = (t.detach().to("cpu").to(F64) for t in self)
        return torch.complex(rh + rl, ih + il)


# ── Conversions ──────────────────────────────────────────────────────────────


def split_fp64(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(hi, lo) float32 of float64 ``x`` (on the host): hi = RN32(x),
    lo = RN32(x - hi). x - hi is exact in float64, so hi + lo is the canonical
    double-word of x and |x - hi - lo| <= 2^-48 |x| (|x| >= 2^-102).
    OverflowError for finite |x| that rounds past the fp32 range."""
    x = x.detach().to("cpu", F64)
    hi = x.to(F32)
    bad = torch.isinf(hi) & torch.isfinite(x)
    if bool(bad.any()):
        v = float(x[bad].flatten()[0])
        raise OverflowError(f"double-word FFT: |x| = {abs(v):.6g} is outside the fp32 range")
    lo = torch.where(torch.isfinite(x), x - hi.to(F64), torch.zeros_like(x)).to(F32)
    return hi, lo


def to_dw(x) -> DWComplex:
    """DWComplex of a complex tensor, a ComplexTensor or a DWComplex.

    float64 / complex128 parts are split on the host (exact as
    ``split_fp64``); float32 parts become (x, 0) on their own device (no
    host round trip); half parts are upcast exactly."""
    from .complex import ComplexTensor

    if isinstance(x, DWComplex):
        return x
    if isinstance(x, ComplexTensor):
        re, im = x.real, x.imag
    elif isinstance(x, torch.Tensor) and x.is_complex():
        re, im = x.real, x.imag
    elif isinstance(x, torch.Tensor):
        re, im = x, torch.zeros_like(x)
    else:
        raise TypeError(f"double-word FFT: cannot convert {type(x).__name__}")
    if re.dtype == F64:
        rh, rl = split_fp64(re)
        ih, il = split_fp64(im)
        return DWComplex(rh, rl, ih, il)
    if re.dtype in (torch.float16, torch.bfloat16, F32):
        re, im = re.to(F32), im.to(F32)
        return DWComplex(re, torch.zeros_like(re), im, torch.zeros_like(im))
    raise TypeError(f"double-word FFT takes floating-point parts; got {re.dtype}")


# ── Error-free transformations and double-word arithmetic ───────────────────
#
# Every function takes an ops object ``o`` with ``add``, ``sub``, ``mul``
# (elementwise fp32, ``out=`` optional), ``mulc`` (times a Python float) and
# ``part(c)``: the ops object for the real (c = 0) or imaginary (c = 1) chain
# of a complex operation (the kernel's ``engines="split"`` puts them on two
# engines; the operations and their order do not change).
# ``_TorchOps`` is the CPU reference; ``trnfft.nki.dw_fft`` supplies one that
# issues one engine instruction per call. The twiddle operand of a product is
# always the second operand (the kernel passes it as a broadcast view).


def _two_sum(o, a, b):
    """Knuth's TwoSum: s + e == a + b exactly. 6 ops."""
    s = o.add(a, b)
    bb = o.sub(s, a)
    t = o.sub(s, bb)
    e1 = o.sub(a, t)
    e2 = o.sub(b, bb)
    return s, o.add(e1, e2)


def _two_diff(o, a, b):
    """TwoSum(a, -b), bit for bit (RN is odd-symmetric): s + e == a - b. 6 ops."""
    s = o.sub(a, b)
    bb = o.sub(s, a)
    t = o.sub(s, bb)
    e1 = o.sub(a, t)
    e2 = o.add(b, bb)
    return s, o.sub(e1, e2)


def _fast_two_sum(o, a, b, out=None):
    """Dekker's FastTwoSum (|a| >= |b| or a == 0): s + e == a + b. 3 ops."""
    s = o.add(a, b, out=None if out is None else out[0])
    z = o.sub(s, a)
    return s, o.sub(b, z, out=None if out is None else out[1])


def _split(o, a):
    """Veltkamp split: a == h + l, each with at most 12 significant bits. 4 ops."""
    c = o.mulc(a, SPLITTER)
    d = o.sub(c, a)
    h = o.sub(c, d)
    return h, o.sub(a, h)


def _dw_add(o, x, y, mode, out=None):
    """x + y for double-words x = (hi, lo), y: AccurateDWPlusDW (Joldes et al.
    2017, Alg. 6, 20 ops, relative error <= 3u^2 + 13u^3) or SloppyDWPlusDW
    (Alg. 5, 11 ops; accurate unless x and y nearly cancel)."""
    sh, sl = _two_sum(o, x[0], y[0])
    if mode == "sloppy":
        v = o.add(x[1], y[1])
        w = o.add(sl, v)
        return _fast_two_sum(o, sh, w, out)
    th, tl = _two_sum(o, x[1], y[1])
    sl = o.add(sl, th)
    vh, vl = _fast_two_sum(o, sh, sl)
    vl = o.add(vl, tl)
    return _fast_two_sum(o, vh, vl, out)


def _dw_sub(o, x, y, mode, out=None):
    """x - y: ``_dw_add(x, -y)`` bit for bit, without the negations."""
    sh, sl = _two_diff(o, x[0], y[0])
    if mode == "sloppy":
        v = o.sub(x[1], y[1])
        w = o.add(sl, v)
        return _fast_two_sum(o, sh, w, out)
    th, tl = _two_diff(o, x[1], y[1])
    sl = o.add(sl, th)
    vh, vl = _fast_two_sum(o, sh, sl)
    vl = o.add(vl, tl)
    return _fast_two_sum(o, vh, vl, out)


def _dw_mul(o, x, xs, y, ys):
    """DWTimesDW without FMA (Joldes et al. 2017, Alg. 10 with Dekker's
    TwoProd), normalised: x = (hi, lo), xs = split(x.hi), likewise y. 16 ops
    given the splits. The error term of TwoProd is unique (p + e == a b), so
    any exact TwoProd gives the same bits."""
    p = o.mul(x[0], y[0])
    t = o.sub(p, o.mul(xs[0], ys[0]))
    t = o.sub(t, o.mul(xs[1], ys[0]))
    t = o.sub(t, o.mul(xs[0], ys[1]))
    e = o.sub(o.mul(xs[1], ys[1]), t)
    tl1 = o.mul(x[0], y[1])
    tl2 = o.mul(x[1], y[0])
    cl2 = o.add(tl1, tl2)
    cl3 = o.add(e, cl2)
    return _fast_two_sum(o, p, cl3)


def _cmul(o, a, w, ws, mode):
    """a * w for double-word complex a = (rh, rl, ih, il) and twiddle w, with
    ws = (split(w.rh), split(w.ih)) precomputed. Each component is a sum of
    two DWTimesDW products (Stloukal and Jezequel, arXiv:2611.08371)."""
    ore, oim = o.part(0), o.part(1)
    ar, ai = (a[0], a[1]), (a[2], a[3])
    wr, wi = (w[0], w[1]), (w[2], w[3])
    ars = _split(ore, a[0])
    ais = _split(oim, a[2])
    re = _dw_sub(ore, _dw_mul(ore, ar, ars, wr, ws[0]), _dw_mul(ore, ai, ais, wi, ws[1]), mode)
    im = _dw_add(oim, _dw_mul(oim, ar, ars, wi, ws[1]), _dw_mul(oim, ai, ais, wr, ws[0]), mode)
    return re + im


def _cadd(o, x, y, mode, out=None):
    re = _dw_add(o.part(0), x[0:2], y[0:2], mode, None if out is None else out[0:2])
    im = _dw_add(o.part(1), x[2:4], y[2:4], mode, None if out is None else out[2:4])
    return re + im


def _csub(o, x, y, mode, out=None):
    re = _dw_sub(o.part(0), x[0:2], y[0:2], mode, None if out is None else out[0:2])
    im = _dw_sub(o.part(1), x[2:4], y[2:4], mode, None if out is None else out[2:4])
    return re + im


def _butterfly(o, a, w, ws, y, mode):
    """One forward Stockham butterfly. ``a``: R double-word complex inputs
    (4-tuples), ``w`` / ``ws``: the R - 1 twiddles and their hi splits (None
    when Ns = 1), ``y``: R output 4-tuples (written through ``out=``)."""
    R = len(a)
    if w is None:
        b = list(a)
    else:
        b = [a[0]] + [_cmul(o, a[r], w[r - 1], ws[r - 1], mode) for r in range(1, R)]
    if R == 2:
        _cadd(o, b[0], b[1], mode, y[0])
        _csub(o, b[0], b[1], mode, y[1])
        return
    t0 = _cadd(o, b[0], b[2], mode)
    t1 = _csub(o, b[0], b[2], mode)
    t2 = _cadd(o, b[1], b[3], mode)
    t3 = _csub(o, b[1], b[3], mode)
    _cadd(o, t0, t2, mode, y[0])
    _csub(o, t0, t2, mode, y[2])
    # (-i) t3 = (t3.im, -t3.re): y1 = t1 + (-i) t3, y3 = t1 - (-i) t3.
    ore, oim = o.part(0), o.part(1)
    _dw_add(ore, t1[0:2], t3[2:4], mode, y[1][0:2])
    _dw_sub(oim, t1[2:4], t3[0:2], mode, y[1][2:4])
    _dw_sub(ore, t1[0:2], t3[2:4], mode, y[3][0:2])
    _dw_add(oim, t1[2:4], t3[0:2], mode, y[3][2:4])


# ── Plan and twiddle table ───────────────────────────────────────────────────


class Stage(NamedTuple):
    R: int  # radix (2 or 4)
    Ns: int  # size of the sub-transforms already done
    G: int  # n / (R Ns)
    off: int  # column offset of this stage's twiddle block in the table (Ns > 1)


def supported(n: int) -> bool:
    """True for the sizes the double-word FFT runs: powers of two, 2..4096."""
    return MIN_N <= n <= MAX_N and n & (n - 1) == 0


def plan(n: int) -> tuple[Stage, ...]:
    """The stage list of length n: radix 2 first when log2(n) is odd, then
    radix 4. Twiddle blocks (Ns > 1) are 12 Ns floats each, in stage order."""
    if not supported(n):
        raise ValueError(f"double-word FFT: n must be a power of two in [2, 4096]; got {n}")
    m = n.bit_length() - 1
    radices = ([2] if m % 2 else []) + [4] * (m // 2)
    stages, Ns, off = [], 1, 0
    for R in radices:
        stages.append(Stage(R, Ns, n // (R * Ns), off))
        if Ns > 1:
            off += 4 * (R - 1) * Ns
        Ns *= R
    return tuple(stages)


def table_size(n: int) -> int:
    """Width of the twiddle table for length n: 4 n floats (the blocks use
    12 sum(Ns) < 4 n; the rest is zero padding), so the kernel reads n from
    the table's shape."""
    plan(n)
    return 4 * n


def _cis(j: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """(cos, sin)(2 pi j / N) in float64 (N a multiple of 4), from the first
    octant: libm is evaluated only on angles in [0, pi/4], so zeros, ones and
    +-sqrt(1/2) are exact (no -0.0) and every symmetry of the circle holds
    bit for bit."""
    m = np.asarray(j, np.int64) % N
    q = (4 * m) // N  # quadrant
    r = m - q * (N // 4)  # 0 <= r < N / 4
    t = np.minimum(r, N // 4 - r)  # octant-reduced index, 0 <= t <= N / 8
    th = (2.0 * np.pi / N) * t.astype(np.float64)
    ct, st = np.cos(th), np.sin(th)
    half = 8 * t == N
    ct = np.where(t == 0, 1.0, np.where(half, np.sqrt(0.5), ct))
    st = np.where(t == 0, 0.0, np.where(half, np.sqrt(0.5), st))
    direct = r == t
    cr, sr = np.where(direct, ct, st), np.where(direct, st, ct)
    cos = np.select([q == 0, q == 1, q == 2, q == 3], [cr, -sr, -cr, sr])
    sin = np.select([q == 0, q == 1, q == 2, q == 3], [sr, cr, -sr, -cr])
    return cos + 0.0, sin + 0.0  # -0.0 -> +0.0


def twiddles_fp64(n: int) -> np.ndarray:
    """The forward twiddles w^(r k) of every stage with Ns > 1, float64, as a
    (4 (R-1) Ns,) block per stage: [r - 1][re, im][k]. For the tests."""
    out = []
    for st in plan(n):
        if st.Ns == 1:
            continue
        N = st.R * st.Ns
        k = np.arange(st.Ns)
        for r in range(1, st.R):
            c, s = _cis(r * k, N)
            out.append(np.stack([c, 0.0 - s]))  # exp(-i theta)
    return np.concatenate([b.ravel() for b in out]) if out else np.zeros(0)


_TABLES: dict[int, torch.Tensor] = {}


def twiddle_table(n: int) -> torch.Tensor:
    """(1, 4 n) float32 host table. Stage block layout (12 Ns floats
    for radix 4): [r - 1 in 0..2][re_hi, im_hi, re_lo, im_lo][k in 0..Ns-1],
    hi = RN32(w), lo = RN32(w - hi) of the float64 twiddle (hi parts first,
    so the kernel splits them with one instruction per op)."""
    t = _TABLES.get(n)
    if t is not None:
        return t
    blocks = []
    for st in plan(n):
        if st.Ns == 1:
            continue
        N = st.R * st.Ns
        k = np.arange(st.Ns)
        rows = []
        for r in range(1, st.R):
            c, s = _cis(r * k, N)
            s = 0.0 - s  # forward: exp(-2 pi i r k / N) (+0.0, not -0.0)
            ch, sh = c.astype(np.float32), s.astype(np.float32)
            cl = (c - ch.astype(np.float64)).astype(np.float32)
            sl = (s - sh.astype(np.float64)).astype(np.float32)
            rows.append(np.stack([ch, sh, cl, sl]))
        blocks.append(np.stack(rows).ravel())
    flat = np.zeros(table_size(n), np.float32)
    if blocks:
        used = np.concatenate(blocks)
        flat[: used.size] = used
    t = torch.from_numpy(np.ascontiguousarray(flat.astype(np.float32)[None, :]))
    _TABLES[n] = t
    return t


def _block_twiddles(blk, hs, R):
    """Twiddle 4-tuples (re_hi, re_lo, im_hi, im_lo) and hi splits of one
    stage from its (3, 4, Ns) block ``blk`` ([r - 1][re_hi, im_hi, re_lo,
    im_lo][k]) and ``hs`` = split(blk[:, 0:2, :])."""
    w = [(blk[r, 0], blk[r, 2], blk[r, 1], blk[r, 3]) for r in range(R - 1)]
    ws = [((hs[0][r, 0], hs[1][r, 0]), (hs[0][r, 1], hs[1][r, 1])) for r in range(R - 1)]
    return w, ws


# ── CPU reference (one torch fp32 op per device instruction) ─────────────────


class _TorchOps:
    """fp32 elementwise ops on CPU tensors: IEEE round to nearest even, no
    fusion (each call is one torch kernel)."""

    @staticmethod
    def _out(r, out):
        if out is None:
            return r
        out.copy_(r)
        return out

    def add(self, a, b, out=None):
        return self._out(torch.add(a, b), out)

    def sub(self, a, b, out=None):
        return self._out(torch.sub(a, b), out)

    def mul(self, a, b, out=None):
        return self._out(torch.mul(a, b), out)

    def mulc(self, a, c, out=None):
        return self._out(torch.mul(a, c), out)

    def part(self, c):
        return self


def reference_fft(x: DWComplex, add: str = "accurate") -> DWComplex:
    """Forward, unnormalised double-word FFT along the last axis, on the host:
    the kernel's fp32 operations one for one. ``x`` parts are float32 (..., n)."""
    if add not in ADD_MODES:
        raise ValueError(f"add must be one of {ADD_MODES}; got {add!r}")
    n = x.shape[-1]
    stages = plan(n)
    lead = x.shape[:-1]
    cur = [t.detach().to("cpu", F32).reshape(-1, n).contiguous() for t in x]
    B = cur[0].shape[0]
    tab = twiddle_table(n)[0]
    o = _TorchOps()
    for st in stages:
        R, Ns, G = st.R, st.Ns, st.G
        src = [c.reshape(B, R * G, Ns) for c in cur]
        nxt = [torch.empty(B, G, R * Ns, dtype=F32) for _ in range(4)]
        a = [tuple(s[:, r * G : (r + 1) * G, :] for s in src) for r in range(R)]
        y = [tuple(d[:, :, q * Ns : (q + 1) * Ns] for d in nxt) for q in range(R)]
        w = ws = None
        if Ns > 1:
            # (R - 1, 4, Ns) block; its (Ns,) rows broadcast over (B, G, Ns).
            blk = tab[st.off : st.off + 4 * (R - 1) * Ns].reshape(R - 1, 4, Ns)
            hs = _split(o, blk[:, 0:2, :])  # the twiddles' hi parts, as the kernel
            w, ws = _block_twiddles(blk, hs, R)
        _butterfly(o, a, w, ws, y, add)
        cur = [d.reshape(B, n) for d in nxt]
    return DWComplex(*(c.reshape(*lead, n) for c in cur))


# ── Public entry points ──────────────────────────────────────────────────────


def _backend(x: DWComplex, backend: str | None) -> str:
    """'reference', 'simulate' or 'nki' for this call."""
    from .nki import dispatch as D

    if backend is not None:
        if backend not in ("reference", "simulate", "nki"):
            raise ValueError(f"backend must be 'reference', 'simulate' or 'nki'; got {backend!r}")
        return backend
    if x.device.type == "neuron":
        return "nki"
    if D._use_nki() and D.HAS_NKI:
        return "simulate" if D._use_simulator() else "nki"
    return "reference"


def fft_dw_raw(
    x: DWComplex,
    inverse: bool = False,
    *,
    add: str = "accurate",
    backend: str | None = None,
    engines: str = "auto",
) -> DWComplex:
    """Double-word FFT along the last axis, DW in and DW out (unnormalised in
    both directions). On the ``nki`` backend the result stays on the device,
    so several transforms can chain without a host round trip.

    ``backend``: None picks the kernel for neuron tensors and under
    ``set_backend("nki")`` / ``"auto"`` with NKI present (``nki.simulate``
    under ``TRNFFT_USE_SIMULATOR=1``), else the CPU reference. All three give
    the same bits (see the module docstring). ``engines``: the kernel's engine
    assignment, "auto" (the compiler's) or "split" (real-part chains on
    VectorE, imaginary on GpSimd; ``add="accurate"`` only), same bits."""
    if add not in ADD_MODES:
        raise ValueError(f"add must be one of {ADD_MODES}; got {add!r}")
    n = x.shape[-1]
    if not supported(n):
        raise ValueError(f"double-word FFT: n must be a power of two in [2, 4096]; got {n}")
    if inverse:
        x = x.swap()
    be = _backend(x, backend)
    if be == "reference":
        y = reference_fft(x, add)
    else:
        from .nki import dw_fft as K

        y = K.run(x, add, simulate=(be == "simulate"), engines=engines)
    return y.swap() if inverse else y


def fft_dw(
    x,
    inverse: bool = False,
    *,
    add: str = "accurate",
    backend: str | None = None,
    engines: str = "auto",
):
    """fp64-class FFT through double-word fp32 butterflies; complex128 result
    on the host, normalised as ``torch.fft`` (1/n on the inverse).

    ``x``: a complex tensor, a ComplexTensor (float64 parts are split exactly
    on the host; float32 parts are taken as exact), or a DWComplex."""
    y = fft_dw_raw(to_dw(x), inverse, add=add, backend=backend, engines=engines)
    z = y.to_complex128()
    if inverse:
        z = z * (1.0 / y.shape[-1])  # a power of two: exact
    return z


def correct_bits(got, ref) -> float:
    """-log2 of the normwise relative error ||got - ref||_2 / ||ref||_2 (53 if
    exact), with ``ref`` a high-precision reference as complex128 or as a pair
    of exact (re, im) error arrays. For the accuracy tables."""
    g = np.asarray(got, np.complex128)
    r = np.asarray(ref, np.complex128)
    err = np.linalg.norm((g - r).ravel()) / max(np.linalg.norm(r.ravel()), 1e-300)
    return 53.0 if err == 0 else min(53.0, -math.log2(err))
