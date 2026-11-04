"""CPU tests of ``trnfft.sht`` (trnfft#148, trnsci/trnsci#111).

References, all fp64 on the CPU:

* a direct quadrature against ``scipy.special.sph_harm_y`` (small L), which
  shares no code with the recurrence in ``trnfft/sht.py``;
* ``ducc0.sht`` and ``SHTns`` when installed (``pip install ducc0 shtns``).

The fp32 tolerance is 1e-6 relative in the spectral l2 norm (see
``docs/api/sht.md`` for the error model: fp32 unit roundoff u = 6e-8 times a
small factor from the table rounding, the K-term PSUM accumulation and the
FFT stages; measured 1-3e-7 at L = 127..1023).
"""

from __future__ import annotations

import gc
import math

import numpy as np
import pytest
import torch

from trnfft import ComplexTensor
from trnfft.sht import (
    SHT,
    TILE,
    PaddedLegendre,
    encode_plan,
    equiangular_grid,
    gauss_legendre_grid,
    legendre_buckets,
    legendre_forward_torch,
    legendre_inverse_torch,
    lm_index,
    m_offset,
    n_alm,
    parity_rows,
    problem_rows,
)

try:
    import ducc0
except ImportError:
    ducc0 = None

try:
    import shtns
except ImportError:
    shtns = None

FP32_TOL = 1e-6


def _random_alm(lmax, nfields, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((nfields, n_alm(lmax))) + 1j * rng.standard_normal(
        (nfields, n_alm(lmax))
    )
    a[:, : lmax + 1] = a[:, : lmax + 1].real  # m = 0: real for a real field
    return a


def _ct(a: np.ndarray, dtype=torch.float64) -> ComplexTensor:
    return ComplexTensor(
        torch.from_numpy(a.real.copy()).to(dtype), torch.from_numpy(a.imag.copy()).to(dtype)
    )


def _np(c: ComplexTensor) -> np.ndarray:
    return c.real.double().numpy() + 1j * c.imag.double().numpy()


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _ducc_synthesis(alm, lmax, geom, nlat, nlon):
    return np.stack(
        [
            ducc0.sht.experimental.synthesis_2d(
                alm=a[None], spin=0, lmax=lmax, geometry=geom, ntheta=nlat, nphi=nlon
            )[0]
            for a in alm
        ]
    )


def _ducc_analysis(f, lmax, geom):
    return np.stack(
        [
            ducc0.sht.experimental.analysis_2d(map=x[None], spin=0, lmax=lmax, geometry=geom)[0]
            for x in f
        ]
    )


# ── Index layout and grids ───────────────────────────────────────────────────


def test_index_layout_matches_triangular_m_major():
    L = 9
    idx = [lm_index(L, l, m) for m in range(L + 1) for l in range(m, L + 1)]
    assert idx == list(range(n_alm(L)))
    for m in range(L + 1):
        assert parity_rows(L, m, 0) + parity_rows(L, m, 1) == L - m + 1
        assert m_offset(L, m) == lm_index(L, m, m)
    with pytest.raises(ValueError):
        lm_index(L, 2, 3)


@pytest.mark.parametrize("n", [1, 2, 7, 64, 513, 1024])
def test_gauss_legendre_grid_integrates_polynomials(n):
    x, w = gauss_legendre_grid(n)
    assert np.all(np.diff(x) < 0)  # north to south
    assert abs(w.sum() - 2.0) < 1e-13
    for deg in range(0, 2 * n, max(1, (2 * n) // 7)):
        exact = 0.0 if deg % 2 else 2.0 / (deg + 1)
        assert abs(np.dot(w, x**deg) - exact) < 1e-13


@pytest.mark.parametrize("n", [2, 3, 8, 65, 1024])
def test_clenshaw_curtis_grid_integrates_polynomials(n):
    x, w = equiangular_grid(n)
    assert x[0] == 1.0 and x[-1] == -1.0
    for deg in range(0, n):
        exact = 0.0 if deg % 2 else 2.0 / (deg + 1)
        assert abs(np.dot(w, x**deg) - exact) < 1e-13


# ── Independent direct-quadrature reference (scipy) ─────────────────────────


@pytest.mark.parametrize("grid", ["gauss", "equiangular"])
def test_matches_direct_quadrature_reference(grid):
    """fp64 synthesis and analysis against a dense evaluation of every Y_lm
    with ``scipy.special.sph_harm_y`` at L = 12 (no shared code)."""
    sp = pytest.importorskip("scipy.special")
    if not hasattr(sp, "sph_harm_y"):
        pytest.skip("scipy >= 1.15 needed for sph_harm_y")
    L = 12
    s = SHT(L, grid=grid, dtype=torch.float64, fft="torch")
    theta = s.theta
    phi = 2 * np.pi * np.arange(s.nlon) / s.nlon
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    alm = _random_alm(L, 2, seed=1)
    f = np.zeros((2, s.nlat, s.nlon))
    for m in range(L + 1):
        for l in range(m, L + 1):
            y = sp.sph_harm_y(l, m, tt, pp)
            c = alm[:, lm_index(L, l, m)]
            term = c[:, None, None] * y[None]
            f += term.real if m == 0 else 2 * term.real
    got = s.inverse(_ct(alm)).numpy()
    assert _rel(got, f) < 1e-13
    # Analysis by the grid's own quadrature: a_lm = sum w * 2pi/nlon * f * conj(Y)
    wq = s.weights[:, None] * (2 * np.pi / s.nlon)
    ref = np.zeros_like(alm)
    for m in range(L + 1):
        for l in range(m, L + 1):
            y = sp.sph_harm_y(l, m, tt, pp)
            ref[:, lm_index(L, l, m)] = np.einsum("tp,ftp->f", wq * np.conj(y), f)
    assert _rel(_np(s.forward(torch.from_numpy(f))), ref) < 1e-13
    assert _rel(ref, alm) < 1e-13  # the quadrature is exact at this band limit


# ── ducc0 / SHTns ────────────────────────────────────────────────────────────


@pytest.mark.skipif(ducc0 is None, reason="ducc0 not installed")
@pytest.mark.parametrize(
    "grid,lmax,nlat",
    [
        ("gauss", 31, None),
        ("gauss", 32, None),  # nlat = 33: equator row
        ("gauss", 31, 40),  # oversampled
        ("equiangular", 31, None),
        ("equiangular", 20, 43),  # odd nlat
        ("gauss", 127, None),
    ],
)
def test_fp64_matches_ducc0(grid, lmax, nlat):
    s = SHT(lmax, nlat=nlat, grid=grid, dtype=torch.float64, fft="torch")
    geom = "GL" if grid == "gauss" else "CC"
    alm = _random_alm(lmax, 3)
    ref_f = _ducc_synthesis(alm, lmax, geom, s.nlat, s.nlon)
    f = s.inverse(_ct(alm))
    assert _rel(f.numpy(), ref_f) < 1e-12
    ref_a = _ducc_analysis(ref_f, lmax, geom)
    assert _rel(_np(s.forward(torch.from_numpy(ref_f))), ref_a) < 1e-12


@pytest.mark.skipif(ducc0 is None, reason="ducc0 not installed")
@pytest.mark.parametrize(
    "lmax,nfields",
    [(31, 8), (127, 4), pytest.param(511, 2, marks=pytest.mark.slow)],
)
@pytest.mark.parametrize("fft", ["torch", "trnfft"])
def test_fp32_within_tolerance_of_ducc0(lmax, nfields, fft):
    """fp32 tables, fp32 FFT and GEMM: forward-coefficient error, synthesis
    error and round trip within 1e-6 (spectral l2) of fp64 ducc0."""
    s = SHT(lmax, fft=fft)
    alm = _random_alm(lmax, nfields, seed=2)
    ref_f = _ducc_synthesis(alm, lmax, "GL", s.nlat, s.nlon)
    a = _np(s.forward(torch.from_numpy(ref_f).float()))
    assert _rel(a, alm) < FP32_TOL
    f = s.inverse(_ct(alm, torch.float32)).double().numpy()
    assert _rel(f, ref_f) < FP32_TOL
    a2 = _np(s.forward(torch.from_numpy(f).float()))
    assert _rel(_np(s.forward(s.inverse(_ct(a, torch.float32)))), alm) < 2 * FP32_TOL
    assert _rel(a2, alm) < 2 * FP32_TOL


@pytest.mark.skipif(shtns is None, reason="SHTns not installed")
def test_fp64_matches_shtns():
    L = 63
    sh = shtns.sht(L, L, 1, shtns.sht_orthonormal)
    s = SHT(L, dtype=torch.float64, fft="torch")
    sh.set_grid(s.nlat, s.nlon, shtns.sht_gauss | shtns.SHT_PHI_CONTIGUOUS, 1e-14)
    assert sh.nlm == n_alm(L)
    # SHTns orders coefficients m-major with l ascending, as trnfft.sht does.
    alm = _random_alm(L, 1, seed=4)[0]
    f_ref = sh.synth(alm.astype(np.complex128))
    f = s.inverse(_ct(alm[None]))[0].numpy()
    assert np.allclose(np.abs(sh.cos_theta), np.abs(s.x))
    if sh.cos_theta[0] < 0:  # SHTns grid ordered south to north
        f_ref = f_ref[::-1]
    assert _rel(f, f_ref) < 1e-12
    a_ref = sh.analys(np.ascontiguousarray(f_ref))
    assert _rel(_np(s.forward(torch.from_numpy(f.copy()[None])))[0], a_ref) < 1e-12


# ── Tables and polar tiles ───────────────────────────────────────────────────


def test_tables_orthonormal_under_quadrature():
    L = 64
    s = SHT(L, dtype=torch.float64, fft="torch", skip_threshold=0.0)
    T = s.table.numpy()
    w = s.weights[: s.H] * 2 * np.pi
    if s.nlat % 2:  # the equator row is its own mirror
        w[-1] *= 0.5
    for m in (0, 1, 33, L):
        o = m_offset(L, m)
        for q in (0, 1):
            r0 = parity_rows(L, m, 0) if q else 0
            k = parity_rows(L, m, q)
            if k == 0:
                continue
            P = T[o + r0 : o + r0 + k]
            # Northern half only: an even/odd product is symmetric, so x2.
            gram = 2 * (P * w) @ P.T
            assert np.abs(gram - np.eye(k)).max() < 1e-12


def test_no_fp64_underflow_at_high_m_near_the_pole():
    """``sin(theta)^m`` underflows fp64 near the pole at m ~ 1000. The table
    starts ``P_mm`` in log space; check it against the closed form
    ``|Y_mm| = sqrt((2m+1)! / 4 pi) / (2^m m!) sin^m theta`` (lgamma), and
    ``P_{m+1,m} = sqrt(2m+3) x P_mm``."""
    from trnfft.sht import legendre_table

    L = 1023
    theta = np.array([0.02, 0.3, 0.6, 1.2, 1.5])
    x = np.cos(theta)
    T = legendre_table(L, x)
    for m in (500, 1000, 1022):
        log_ref = (
            -m * math.log(2.0)
            - math.lgamma(m + 1)
            + 0.5 * (math.lgamma(2 * m + 2) - math.log(4 * math.pi))
            + m * np.log(np.sin(theta))
        )
        got = T[m_offset(L, m)]
        ok = log_ref > -700
        assert ok.sum() >= 2
        assert np.allclose(np.log(np.abs(got[ok])), log_ref[ok], rtol=0, atol=1e-10)
        assert np.all(np.sign(got[ok]) == (-1) ** m)  # Condon-Shortley
        assert np.all(got[log_ref < -800] == 0.0)  # true underflow, below fp64 range
        p1 = T[m_offset(L, m) + parity_rows(L, m, 0)]  # l = m + 1
        assert np.allclose(p1, math.sqrt(2 * m + 3) * x * got, rtol=1e-13, atol=0)


@pytest.mark.slow
def test_high_m_orthonormal_at_l1023():
    """Orthonormality of the m = 1000 and m = 1023 tables at L = 1023 under
    the Gauss quadrature: the recurrence from the log-space start is right
    all the way to l = L."""
    L = 1023
    s = SHT(L, dtype=torch.float64, fft="torch", skip_threshold=0.0)
    T = s.table.numpy()
    w = s.weights[: s.H] * 2 * np.pi
    for m in (1000, 1023):
        o = m_offset(L, m)
        for q in (0, 1):
            k = parity_rows(L, m, q)
            if k == 0:
                continue
            r0 = parity_rows(L, m, 0) if q else 0
            P = T[o + r0 : o + r0 + k]
            assert np.abs(2 * (P * w) @ P.T - np.eye(k)).max() < 1e-12


def test_polar_mask_marks_exactly_the_zero_tiles():
    L = 511
    s = SHT(L, fft="torch")
    T = s.table.numpy()
    done, total = s.tile_counts
    assert 0 < total - done < total  # some, not all, tiles skipped at L = 511
    ht = -(-s.H // TILE)
    for m in range(0, L + 1, 37):
        o = m_offset(L, m)
        for q in (0, 1):
            r0 = o + (parity_rows(L, m, 0) if q else 0)
            k = parity_rows(L, m, q)
            for kt in range(-(-k // TILE)):
                for h in range(ht):
                    t = T[r0 + kt * TILE : r0 + min(k, (kt + 1) * TILE), h * TILE : (h + 1) * TILE]
                    assert s.tile_mask[2 * m + q, kt, h] == bool(np.any(t != 0))
    # Skipped tiles sit at the pole end (latitude tile 0) for large m.
    assert not s.tile_mask[2 * L, 0, 0]
    assert s.tile_mask[0].all()


def test_dropped_entries_are_below_threshold():
    """The polar drop only removes entries below threshold x per-m max, so it
    changes the transform by far less than fp32 rounding."""
    L = 255
    s0 = SHT(L, dtype=torch.float64, fft="torch", skip_threshold=0.0)
    s1 = SHT(L, dtype=torch.float64, fft="torch")
    d = (s0.table - s1.table).abs()
    for m in range(0, L + 1, 15):
        o = m_offset(L, m)
        blk = s0.table[o : o + L - m + 1].abs()
        assert d[o : o + L - m + 1].max() <= 1e-30 * blk.max()


def test_plan_string_encodes_mask():
    from trnfft.nki.sht import _chunks, _decode_plan, chunk_plan

    s = SHT(300, fft="torch")
    L, H, m0, m1, bits = _decode_plan(s.plan)
    assert (L, H, m0, m1) == (300, s.H, 0, 301)
    assert len(bits) == 2 * (L + 1)
    assert bits[2 * L + 1] == ""  # m = L has no odd-parity degree
    mask = s.tile_mask.copy()
    mask[5, 0, 1] = False
    b2 = _decode_plan(encode_plan(L, s.H, mask))[4]
    assert b2[5][1] == "0" and bits[5][1] == "1"
    assert _decode_plan(chunk_plan(s.plan, 10, 20))[2:4] == (10, 20)
    ch = _chunks(s.plan)
    assert ch[0][0] == 0 and ch[-1][1] == L + 1
    assert all(a[1] == b[0] for a, b in zip(ch, ch[1:], strict=False))


# ── Stage structure ──────────────────────────────────────────────────────────


def test_per_m_gemm_matches_dense_transform():
    """The parity-split per-m loop equals a dense Y matrix product."""
    L = 20
    s = SHT(L, dtype=torch.float64, fft="torch", skip_threshold=0.0)
    from trnfft.sht import legendre_table

    full = legendre_table(L, s.x)  # (n_alm, nlat), all latitudes
    g = torch.Generator().manual_seed(0)
    a = torch.randn(s.n_alm, 6, generator=g, dtype=torch.float64)
    y = legendre_inverse_torch(s.table, a, L, s.H).reshape(L + 1, 2, s.H, 6)
    north = (y[:, 0] + y[:, 1]).numpy()
    for m in range(L + 1):
        o = m_offset(L, m)
        ke = parity_rows(L, m, 0)
        # rows of the packed table -> alm rows (l order)
        ls_even = list(range(m, L + 1, 2))
        ls_odd = list(range(m + 1, L + 1, 2))
        P = np.concatenate([full[o : o + ke], full[o + ke : o + L - m + 1]])
        A = np.concatenate(
            [a[[lm_index(L, l, m) for l in ls_even]], a[[lm_index(L, l, m) for l in ls_odd]]]
        )
        dense = P.T @ A  # (nlat, 6)
        assert np.allclose(north[m], dense[: s.H], atol=1e-12)
    x = torch.randn(2 * (L + 1) * s.H, 6, generator=g, dtype=torch.float64)
    out = legendre_forward_torch(s.table, x, L, s.H)
    for m in (0, 7, L):
        for q in (0, 1):
            p = 2 * m + q
            ls = list(range(m + q, L + 1, 2))
            o = m_offset(L, m) + (parity_rows(L, m, 0) if q else 0)
            ref = s.table[o : o + len(ls)] @ x[p * s.H : (p + 1) * s.H]
            assert torch.allclose(out[[lm_index(L, l, m) for l in ls]], ref)


def test_leading_dims_and_shapes():
    L = 15
    s = SHT(L, fft="torch")
    f = torch.randn(2, 3, s.nlat, s.nlon)
    a = s.forward(f)
    assert a.real.shape == (2, 3, n_alm(L))
    assert s.inverse(a).shape == (2, 3, s.nlat, s.nlon)
    assert s(f).real.shape == (2, 3, n_alm(L))
    with pytest.raises(ValueError):
        s.forward(torch.randn(3, s.nlat + 1, s.nlon))
    with pytest.raises(ValueError):
        s.inverse(ComplexTensor(torch.zeros(3, 5), torch.zeros(3, 5)))


def test_accepts_torch_complex_coefficients():
    L = 10
    s = SHT(L, dtype=torch.float64, fft="torch")
    alm = _random_alm(L, 2)
    f1 = s.inverse(torch.from_numpy(alm))
    f2 = s.inverse(_ct(alm))
    assert torch.equal(f1, f2)


def test_trnfft_and_torch_fft_agree():
    L = 63
    s1 = SHT(L, dtype=torch.float64, fft="torch")
    s2 = SHT(L, dtype=torch.float64, fft="trnfft")
    alm = _ct(_random_alm(L, 2))
    f1, f2 = s1.inverse(alm), s2.inverse(alm)
    assert ((f1 - f2).norm() / f1.norm()).item() < 1e-12
    a1, a2 = s1.forward(f1), s2.forward(f1)
    assert _rel(_np(a2), _np(a1)) < 1e-12


def test_flops_and_tile_counts():
    L = 511
    s = SHT(L, fft="torch")
    full = SHT(L, fft="torch", skip_threshold=0.0)
    F = 128
    # Unskipped flops: 2 * sum_p K_p * H * 2F = 2 * n_alm * H * 2F.
    assert full.flops(F) == 2 * n_alm(L) * s.H * 2 * F
    assert s.flops(F) < full.flops(F)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lmax": -1},
        {"lmax": 8, "grid": "healpix"},
        {"lmax": 8, "nlat": 8},
        {"lmax": 8, "grid": "equiangular", "nlat": 16},
        {"lmax": 8, "nlon": 16},
        {"lmax": 8, "legendre": "cuda"},
        {"lmax": 8, "fft": "fftw"},
        {"lmax": 8, "dtype": torch.float16},
        {"lmax": 8, "legendre": "torch-padded", "buckets": 0},
    ],
)
def test_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        SHT(**kwargs)


def test_legendre_torch_backend_is_default_on_cpu():
    s = SHT(8, fft="torch")
    assert not s._use_nki_legendre() or __import__("trnfft").HAS_NKI


# ── Padded latitude stage (the device-aten-pad baseline) ────────────────────


@pytest.mark.parametrize("lmax", [0, 1, 2, 7, 40, 127, 300])
@pytest.mark.parametrize("buckets", [1, 3, 10, "pow2"])
def test_buckets_cover_live_problems_in_order(lmax, buckets):
    K = problem_rows(lmax)
    assert np.all(np.diff(K) <= 0)  # non-increasing: buckets are p ranges
    bk = legendre_buckets(lmax, buckets)
    live = int((K > 0).sum())
    assert bk[0][0] == 0 and bk[-1][1] == live
    for (p0, p1, kb), nxt in zip(bk, bk[1:] + [(live, None, None)], strict=True):
        assert p0 < p1 == nxt[0]
        assert kb == K[p0] if buckets != "pow2" else kb >= K[p0]
    if buckets != "pow2":
        assert len(bk) == min(buckets, len(set(K[:live].tolist())))


def test_optimal_buckets_beat_pow2_and_shrink_with_b():
    K = problem_rows(511)

    def cost(bk):
        return sum((p1 - p0) * kb for p0, p1, kb in bk) / K.sum()

    c = [cost(legendre_buckets(511, b)) for b in (4, 8, 10, 12)]
    assert all(a > b for a, b in zip(c, c[1:], strict=False))
    assert c[2] < 1.10 < cost(legendre_buckets(511, "pow2"))


@pytest.mark.parametrize(
    "lmax,nlat,grid",
    [
        (0, None, "gauss"),
        (7, None, "gauss"),
        (40, 43, "gauss"),
        (63, None, "equiangular"),
        (127, None, "gauss"),
    ],
)
@pytest.mark.parametrize("buckets", [1, 10, "pow2"])
def test_padded_latitude_stage_equals_per_m_loop(lmax, nlat, grid, buckets):
    s = SHT(lmax, nlat=nlat, grid=grid, dtype=torch.float64, fft="torch")
    pad = PaddedLegendre(s.table, lmax, s.H, buckets)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2 * (lmax + 1) * s.H, 6, dtype=torch.float64, generator=g)
    a = torch.randn(s.n_alm, 6, dtype=torch.float64, generator=g)
    ref_a = legendre_forward_torch(s.table, x, lmax, s.H)
    ref_y = legendre_inverse_torch(s.table, a, lmax, s.H)
    assert torch.allclose(pad.forward(x), ref_a, rtol=0, atol=1e-12 * ref_a.abs().max())
    assert torch.allclose(pad.inverse(a), ref_y, rtol=0, atol=1e-12 * ref_y.abs().max())
    assert pad.flops(1) >= sum(2 * k * s.H * 2 for k in problem_rows(lmax))


@pytest.mark.skipif(ducc0 is None, reason="ducc0 not installed")
@pytest.mark.parametrize(
    "lmax,nfields",
    [(31, 8), (127, 4), pytest.param(511, 2, marks=pytest.mark.slow)],
)
@pytest.mark.parametrize("buckets", [10, "pow2"])
def test_padded_fp32_within_tolerance_of_ducc0(lmax, nfields, buckets):
    """legendre="torch-padded" in fp32: within 1e-6 of fp64 ducc0 and within
    fp32 summation-order noise of the per-m loop."""
    s = SHT(lmax, fft="torch", legendre="torch-padded", buckets=buckets)
    u = SHT(lmax, fft="torch", legendre="torch")
    alm = _random_alm(lmax, nfields, seed=2)
    ref_f = _ducc_synthesis(alm, lmax, "GL", s.nlat, s.nlon)
    f32 = torch.from_numpy(ref_f).float()
    a = _np(s.forward(f32))
    assert _rel(a, alm) < FP32_TOL
    assert _rel(a, _np(u.forward(f32))) < FP32_TOL / 4
    f = s.inverse(_ct(alm, torch.float32)).double().numpy()
    assert _rel(f, ref_f) < FP32_TOL
    assert _rel(f, u.inverse(_ct(alm, torch.float32)).double().numpy()) < FP32_TOL / 4
    assert _rel(_np(s.forward(s.inverse(_ct(alm, torch.float32)))), alm) < 2 * FP32_TOL


# ── On hardware ──────────────────────────────────────────────────────────────


@pytest.mark.neuron
@pytest.mark.parametrize("lmax,nfields", [(127, 16), (511, 128), (1023, 128)])
def test_device_kernels_match_fp64(lmax, nfields):
    """NKI latitude kernels on the device (and trnfft FFTs) within 1e-6 of fp64."""
    import trnfft

    if ducc0 is None:
        pytest.skip("ducc0 not installed")
    trnfft.set_backend("nki")
    try:
        s = SHT(lmax, legendre="nki", device="neuron")
        alm = _random_alm(lmax, nfields, seed=5)
        ref_f = _ducc_synthesis(alm, lmax, "GL", s.nlat, s.nlon)
        x = torch.from_numpy(ref_f).float().to("neuron")
        a = s.forward(x)
        assert _rel(_np(ComplexTensor(a.real.cpu(), a.imag.cpu())), alm) < FP32_TOL
        # At LNC=1, L = 1023 the forward's operands and the allocator blocks
        # they leave cached (14 GiB reserved) plus the inverse's 13 GiB peak
        # exceed one physical core's HBM (NRT_RESOURCE on a 1 GiB alloc), so
        # release them first (trnfft#152, #156).
        del x, a
        gc.collect()
        empty_cache = getattr(getattr(torch, "neuron", None), "empty_cache", None)
        if empty_cache is not None:
            empty_cache()
        re = torch.from_numpy(alm.real.copy()).float().to("neuron")
        im = torch.from_numpy(alm.imag.copy()).float().to("neuron")
        f = s.inverse(ComplexTensor(re, im))
        assert _rel(f.cpu().double().numpy(), ref_f) < FP32_TOL
    finally:
        trnfft.set_backend("auto")
