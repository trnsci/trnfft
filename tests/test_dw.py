"""Double-word FFT (trnfft#172), CPU: error-free building blocks, twiddles,
plan and layout, accuracy against mpmath, determinism, and the
``precision="dw"`` route. The kernel's own checks (MLIR verify, simulator
bitwise) are in ``test_dw_nki.py``; device checks in ``test_dw_neuron.py``."""

from __future__ import annotations

import math
from fractions import Fraction

import mpmath
import numpy as np
import pytest
import torch

import trnfft
from trnfft import dw
from trnfft.complex import ComplexTensor
from trnfft.fft_core import fft_core
from trnfft.nki import dw_fft as K

O = dw._TorchOps()
SIZES = [1 << m for m in range(1, 13)]


def _f32(rng, shape, spread=8):
    x = rng.standard_normal(shape) * 2.0 ** rng.integers(-spread, spread, shape)
    return torch.from_numpy(x.astype(np.float32))


def _exact(t):
    return [Fraction(float(v)) for v in t.flatten().tolist()]


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


# ── Error-free transformations ───────────────────────────────────────────────


class TestEFT:
    def test_two_sum_and_two_diff_exact(self):
        rng = np.random.default_rng(1)
        a, b = _f32(rng, 4000, 30), _f32(rng, 4000, 30)
        s, e = dw._two_sum(O, a, b)
        for x, y, u, v in zip(_exact(a), _exact(b), _exact(s), _exact(e), strict=True):
            assert u + v == x + y
        d, f = dw._two_diff(O, a, b)
        s2, e2 = dw._two_sum(O, a, -b)
        assert _bits_equal(d, s2) and _bits_equal(f, e2)  # TwoSum(a, -b) bit for bit

    def test_split_and_two_prod_exact(self):
        rng = np.random.default_rng(2)
        a, b = _f32(rng, 4000, 40), _f32(rng, 4000, 40)
        h, l = dw._split(O, a)
        assert torch.equal(h + l, a)
        ah, bh = dw._split(O, a), dw._split(O, b)
        z = torch.zeros_like(a)
        p, e = dw._dw_mul(O, (a, z), ah, (b, z), bh)  # lo = 0: TwoProd then FastTwoSum
        for x, y, u, v in zip(_exact(a), _exact(b), _exact(p), _exact(e), strict=True):
            assert u + v == x * y

    @pytest.mark.parametrize("mode", dw.ADD_MODES)
    def test_dw_add_sub_mul_error_bounds(self, mode):
        rng = np.random.default_rng(3)
        x = dw.split_fp64(torch.from_numpy(rng.standard_normal(3000)))
        y = dw.split_fp64(torch.from_numpy(rng.standard_normal(3000)))
        X = [a + b for a, b in zip(_exact(x[0]), _exact(x[1]), strict=True)]
        Y = [a + b for a, b in zip(_exact(y[0]), _exact(y[1]), strict=True)]
        s = dw._dw_add(O, x, y, mode)
        d = dw._dw_sub(O, x, y, mode)
        m = dw._dw_mul(O, x, dw._split(O, x[0]), y, dw._split(O, y[0]))
        # Joldes et al. 2017: accurate add 3u^2 + 13u^3, DWTimesDW (no FMA) 7u^2;
        # sloppy add is unbounded under cancellation, so bound it by |x| + |y|.
        u2 = Fraction(1, 2**48)
        for i in range(3000):
            for got, exact, scale, bound in (
                (s, X[i] + Y[i], abs(X[i]) + abs(Y[i]), 4 * u2),
                (d, X[i] - Y[i], abs(X[i]) + abs(Y[i]), 4 * u2),
                (m, X[i] * Y[i], abs(X[i] * Y[i]), 8 * u2),
            ):
                v = Fraction(float(got[0][i])) + Fraction(float(got[1][i]))
                ref = scale if (mode == "sloppy" and got is not m) else abs(exact)
                assert abs(v - exact) <= bound * ref
                # normalised: |lo| <= ulp32(hi) / 2
                assert abs(float(got[1][i])) <= float(np.spacing(np.abs(got[0][i].numpy()))) / 2

    def test_dw_sub_is_add_of_negation(self):
        rng = np.random.default_rng(4)
        x = dw.split_fp64(torch.from_numpy(rng.standard_normal(2000)))
        y = dw.split_fp64(torch.from_numpy(rng.standard_normal(2000)))
        for mode in dw.ADD_MODES:
            a = dw._dw_sub(O, x, y, mode)
            b = dw._dw_add(O, x, (-y[0], -y[1]), mode)
            assert _bits_equal(a[0], b[0]) and _bits_equal(a[1], b[1])

    def test_matches_trnblas_fp64(self):
        """Vendored primitives give the bits of trnblas.fp64 (when installed)."""
        F = pytest.importorskip("trnblas.fp64")
        rng = np.random.default_rng(5)
        x = dw.split_fp64(torch.from_numpy(rng.standard_normal(2000)))
        y = dw.split_fp64(torch.from_numpy(rng.standard_normal(2000)))
        pairs = (
            (dw._dw_add(O, x, y, "accurate"), F.ff_add(x, y)),
            (dw._dw_add(O, x, y, "sloppy"), F.ff_add_sloppy(x, y)),
            (dw._dw_mul(O, x, dw._split(O, x[0]), y, dw._split(O, y[0])), F.ff_mul(x, y)),
            (dw._two_sum(O, x[0], y[0]), F.two_sum(x[0], y[0])),
        )
        for ours, theirs in pairs:
            assert _bits_equal(ours[0], theirs[0]) and _bits_equal(ours[1], theirs[1])


# ── Twiddles, plan, layout ───────────────────────────────────────────────────


class TestTables:
    @pytest.mark.parametrize("n", SIZES)
    def test_plan(self, n):
        st = dw.plan(n)
        assert math.prod(s.R for s in st) == n
        assert all(s.R * s.Ns * s.G == n for s in st)
        assert st[0].Ns == 1 and (st[0].R == 2) == (n.bit_length() % 2 == 0)
        used = sum(4 * (s.R - 1) * s.Ns for s in st if s.Ns > 1)
        assert used < dw.table_size(n) == 4 * n

    @pytest.mark.parametrize("n", [0, 1, 3, 6, 8192])
    def test_unsupported(self, n):
        assert not dw.supported(n)
        with pytest.raises(ValueError):
            dw.plan(n)

    def test_cis_exact_and_accurate(self):
        for N in (4, 8, 64, 4096, 16384):
            j = np.arange(N)
            c, s = dw._cis(j, N)
            with mpmath.workprec(120):
                for k in range(0, N, max(1, N // 257)):
                    ang = 2 * mpmath.pi * k / N
                    # libm plus the rounded angle: ~1 ulp, 16x under u_dw = 2^-48
                    assert abs(c[k] - mpmath.cos(ang)) <= 2**-52
                    assert abs(s[k] - mpmath.sin(ang)) <= 2**-52
            q = N // 4
            assert (c[0], s[0], c[q], s[q], c[2 * q], s[2 * q]) == (1, 0, 0, 1, -1, 0)
            assert np.array_equal(c[1:], c[1:][::-1]) and np.array_equal(s[1:], -s[1:][::-1])
            assert not np.signbit(c[c == 0]).any() and not np.signbit(s[s == 0]).any()
            # quarter-turn symmetry, bit for bit
            assert np.array_equal(c, np.roll(s, -q))
            if N >= 8:
                assert c[N // 8] == s[N // 8]

    @pytest.mark.parametrize("n", [8, 64, 4096])
    def test_table_is_split_of_fp64_twiddles(self, n):
        t = dw.twiddle_table(n)[0].numpy()
        w = dw.twiddles_fp64(n)
        off = 0
        for st in dw.plan(n):
            if st.Ns == 1:
                continue
            blk = t[st.off : st.off + 12 * st.Ns].reshape(3, 4, st.Ns).astype(np.float64)
            ref = w[off : off + 6 * st.Ns].reshape(3, 2, st.Ns)
            off += 6 * st.Ns
            for r in range(3):
                for c, (h, lo) in enumerate(((0, 2), (1, 3))):
                    assert (np.abs(blk[r, h] + blk[r, lo] - ref[r, c]) <= 2**-49).all()
                    assert np.array_equal(blk[r, h], ref[r, c].astype(np.float32))

    @pytest.mark.parametrize("nprog", [1, 2])
    @pytest.mark.parametrize("B", [1, 2, 3, 127, 128, 129, 300, 1000, 5000])
    @pytest.mark.parametrize("n", [2, 4, 64, 512, 2048, 4096])
    def test_layout(self, n, B, nprog):
        Rb, P, NT = K.layout(B, n, nprog)
        assert Rb & (Rb - 1) == 0 and Rb * n <= max(n, K.F_TARGET)
        assert 1 <= P <= 128 and NT * P * Rb >= B
        assert NT * P * Rb - B < P * Rb * max(1, nprog)  # padding under one tile per program
        if nprog > 1 and -(-B // Rb) >= nprog:
            assert NT % nprog == 0
        assert P == 128 or NT * P <= 128  # no tile straddles partition row 128
        # the kernel's own tiling of the packed rows matches, and at LNC=2 the
        # programs split the tiles (no duplicated work) whenever M >= 2
        M = NT * P
        assert K.tiles(M, nprog) == (P, NT, nprog if M >= nprog else 1)
        F = Rb * n
        rc, gc, kc = K.chunks(Rb, 1, F, K.chunk_width(F))
        assert rc * gc * kc <= K.chunk_width(F)

    @pytest.mark.parametrize("n", SIZES)
    def test_chunks_tile_every_stage(self, n):
        for F in {n, max(n, K.F_TARGET)}:
            Rb = F // n
            for st in dw.plan(n):
                rc, gc, kc = K.chunks(Rb, st.G, st.Ns, K.chunk_width(F))
                assert Rb % rc == 0 and st.G % gc == 0 and st.Ns % kc == 0
                assert rc * gc * kc <= K.chunk_width(F) and st.Ns // kc <= 4


# ── Accuracy against mpmath ──────────────────────────────────────────────────


def _mp_dft(row, prec=160):
    """Exact-enough DFT (unnormalised) of one row by an mpmath radix-2 FFT."""
    with mpmath.workprec(prec):
        n = len(row)
        m = n.bit_length() - 1
        a = [mpmath.mpc(v.real, v.imag) for v in row]
        a = [a[int(bin(i)[2:].zfill(m)[::-1], 2)] for i in range(n)]
        L = 2
        while L <= n:
            h = L // 2
            ws = [mpmath.expjpi(-mpmath.mpf(2 * k) / L) for k in range(h)]
            for s in range(0, n, L):
                for k in range(h):
                    u, v = a[s + k], a[s + k + h] * ws[k]
                    a[s + k], a[s + k + h] = u + v, u - v
            L *= 2
        return a


def _bits(got, ref):
    with mpmath.workprec(160):
        num = mpmath.fsum(
            abs(mpmath.mpc(g.real, g.imag) - r) ** 2 for g, r in zip(got, ref, strict=True)
        )
        den = mpmath.fsum(abs(r) ** 2 for r in ref)
        return 99.0 if num == 0 else float(-mpmath.log(mpmath.sqrt(num / den), 2))


# Floors 1.2-2 bits under the measured CPU table (trnfft#172, worst of 8 rows):
# "accurate" 48.9 bits at N=4, 48.0 at N=8, 47.5 at N=4096.
_FLOOR = {n: 47.0 - 0.1 * (n.bit_length() - 1) for n in SIZES}


@pytest.mark.parametrize("mode", dw.ADD_MODES)
@pytest.mark.parametrize("n", [4, 16, 64, 256, 1024, 4096])
def test_accuracy_vs_mpmath(n, mode):
    rng = np.random.default_rng(n)
    x = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
    d = dw.to_dw(torch.from_numpy(x))
    xe = d.to_complex128().numpy()  # hi + lo: exact in float64
    y = dw.fft_dw_raw(d, add=mode, backend="reference").to_complex128().numpy()
    host = torch.fft.fft(torch.from_numpy(xe)).numpy()
    for i in range(2):
        ref = _mp_dft(xe[i])
        b = _bits(y[i], ref)
        assert b >= _FLOOR[n] - (0.5 if mode == "sloppy" else 0.0), (n, mode, b)
        assert b <= _bits(host[i], ref) + 0.5  # not better than fp64 (sanity of the metric)
        assert b >= _bits(host[i], ref) - 6.0  # measured: 4.1-5.1 bits under host fp64


@pytest.mark.parametrize("n", SIZES)
def test_matches_numpy_and_inverts(n):
    rng = np.random.default_rng(10 + n)
    x = rng.standard_normal((3, n)) + 1j * rng.standard_normal((3, n))
    X = dw.fft_dw(torch.from_numpy(x)).numpy()
    ref = np.fft.fft(x)
    assert np.abs(X - ref).max() <= 1e-13 * np.abs(ref).max()
    back = dw.fft_dw(torch.from_numpy(ref), inverse=True).numpy()
    assert np.abs(back - x).max() <= 1e-13 * np.abs(x).max()


def test_inverse_is_swapped_forward_bitwise():
    rng = np.random.default_rng(7)
    d = dw.to_dw(
        torch.from_numpy(rng.standard_normal((2, 256)) + 1j * rng.standard_normal((2, 256)))
    )
    inv = dw.fft_dw_raw(d, inverse=True, backend="reference")
    fwd = dw.reference_fft(d.swap()).swap()
    assert all(_bits_equal(a, b) for a, b in zip(inv, fwd, strict=True))


def test_special_inputs_exact():
    n = 64
    z = torch.zeros(n, dtype=torch.complex128)
    assert torch.equal(dw.fft_dw(z), z)
    imp = z.clone()
    imp[0] = 1
    assert torch.equal(dw.fft_dw(imp), torch.ones(n, dtype=torch.complex128))
    const = torch.ones(n, dtype=torch.complex128)
    exp = torch.zeros(n, dtype=torch.complex128)
    exp[0] = n
    assert torch.equal(dw.fft_dw(const), exp)


def test_overflow_raises():
    with pytest.raises(OverflowError):
        dw.fft_dw(torch.tensor([1e39, 0.0], dtype=torch.complex128))


# ── Determinism: one fixed order per n ───────────────────────────────────────


@pytest.mark.parametrize("mode", dw.ADD_MODES)
@pytest.mark.parametrize("n", [8, 512, 4096])
def test_rows_are_independent_bitwise(n, mode):
    """Each row's bits depend only on that row: batch size, position and the
    kernel layout (Rb, P, NT, padding, LNC split) cannot change them."""
    rng = np.random.default_rng(n)
    d = dw.to_dw(torch.from_numpy(rng.standard_normal((5, n)) + 1j * rng.standard_normal((5, n))))
    full = dw.reference_fft(d, mode)
    for i in (0, 3):
        one = dw.reference_fft(dw.DWComplex(*(t[i : i + 1] for t in d)), mode)
        assert all(_bits_equal(a[i : i + 1], b) for a, b in zip(full, one, strict=True))
    planes, B = K.pack(d, 2)
    assert B == 5 and planes[0].shape[1] % n == 0
    rows = dw.DWComplex(*(p.reshape(-1, n) for p in planes))
    padded = dw.reference_fft(rows, mode)
    assert all(_bits_equal(a, b[:5]) for a, b in zip(full, padded, strict=True))
    again = dw.reference_fft(d, mode)
    assert all(_bits_equal(a, b) for a, b in zip(full, again, strict=True))


# ── The precision="dw" route ─────────────────────────────────────────────────


@pytest.fixture
def pytorch_backend():
    prev = trnfft.get_backend()
    trnfft.set_backend("pytorch")
    yield
    trnfft.set_backend(prev)
    trnfft.set_precision("fast")


def test_precision_dw_float64(pytorch_backend):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 1024)) + 1j * rng.standard_normal((4, 1024))
    ct = ComplexTensor(torch.from_numpy(x.real.copy()), torch.from_numpy(x.imag.copy()))
    y = fft_core(ct, precision="dw")
    assert y.real.dtype == torch.float64
    ref = np.fft.fft(x)
    got = y.real.numpy() + 1j * y.imag.numpy()
    assert np.abs(got - ref).max() <= 1e-13 * np.abs(ref).max()
    trnfft.set_precision("dw")
    y2 = trnfft.fft(torch.from_numpy(x))
    assert torch.equal(y2.real, y.real) and torch.equal(y2.imag, y.imag)
    yi = trnfft.ifft(torch.from_numpy(ref))
    assert np.abs(yi.real.numpy() + 1j * yi.imag.numpy() - x).max() <= 1e-13


def test_precision_dw_float32_returns_float32(pytorch_backend):
    x = torch.randn(3, 256, generator=torch.Generator().manual_seed(0))
    y = fft_core(ComplexTensor(x, torch.zeros_like(x)), precision="dw")
    assert y.real.dtype == torch.float32
    ref = torch.fft.fft(x.double())
    assert torch.equal(y.real, ref.real.float()) or (y.real - ref.real.float()).abs().max() <= (
        ref.real.abs().max() * 2**-23
    )


def test_precision_dw_other_sizes_take_double(pytorch_backend):
    x = torch.randn(2, 100, dtype=torch.float64)
    ct = ComplexTensor(x, torch.zeros_like(x))
    a = fft_core(ct, precision="dw")
    b = fft_core(ct, precision="double")
    assert torch.equal(a.real, b.real) and torch.equal(a.imag, b.imag)


def test_precision_dw_refuses_autograd(pytorch_backend):
    x = torch.randn(64, dtype=torch.float64, requires_grad=True)
    with pytest.raises(NotImplementedError):
        fft_core(ComplexTensor(x, torch.zeros_like(x)), precision="dw")
