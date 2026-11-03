"""``nki.simulate`` tests of the SHT latitude kernels (trnfft#148).

Run with ``TRNFFT_USE_SIMULATOR=1`` on an x86_64 Linux host with ``nki``.
Each kernel is checked against the per-m torch GEMM loop in fp64 on the same
fp32 table, at one program and at two (``kernel[2]``: the LNC=2 split of the
columns over both physical cores via ``program_id``), with tail tiles in every
dimension, odd ``nlat``, polar-tile masks and chunked launches. The MLIR
verifier runs in ``tests/test_nki_compile.py`` (``sht-*`` cases).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nki_simulator


@pytest.fixture(autouse=True)
def _simulator_enabled():
    if os.environ.get("TRNFFT_USE_SIMULATOR", "").lower() not in ("1", "true", "yes"):
        pytest.skip("TRNFFT_USE_SIMULATOR=1 required")
    from trnfft.nki.dispatch import HAS_NKI

    if not HAS_NKI:
        pytest.skip("nki package not importable on this host")


@pytest.fixture(params=[1, 2], ids=["programs1", "programs2"])
def programs(request, monkeypatch):
    monkeypatch.setenv("TRNFFT_SHT_SIM_PROGRAMS", str(request.param))
    return request.param


def _operands(s, nfields, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = s.lmax
    x = torch.randn(2 * (L + 1) * s.H, 2 * nfields, generator=g)
    a = torch.randn(s.n_alm, 2 * nfields, generator=g)
    return x, a


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _check(s, nfields, table_for_kernel=None, seed=0):
    from trnfft.nki.sht import legendre_forward_nki, legendre_inverse_nki
    from trnfft.sht import legendre_forward_torch, legendre_inverse_torch

    L, H = s.lmax, s.H
    x, a = _operands(s, nfields, seed)
    t = s.table if table_for_kernel is None else table_for_kernel
    ref_f = legendre_forward_torch(s.table.double(), x.double(), L, H)
    ref_i = legendre_inverse_torch(s.table.double(), a.double(), L, H)
    out_f = legendre_forward_nki(t, x, L, H, s.plan)
    out_i = legendre_inverse_nki(t, a, L, H, s.plan)
    assert out_f.shape == ref_f.shape and out_i.shape == ref_i.shape
    assert torch.isfinite(out_f).all() and torch.isfinite(out_i).all()
    # fp32 inputs and products, fp32 PSUM accumulation over K <= 2 * 128.
    assert _rel(out_f, ref_f) < 1e-6
    assert _rel(out_i, ref_i) < 1e-6


@pytest.mark.parametrize(
    "lmax,nfields",
    [
        (0, 1),  # one coefficient, one latitude row
        (7, 2),  # H = 4, every dimension a tail
        (40, 3),  # nlat = 41 odd: equator row
        (130, 5),  # K_e(0) = 66, H = 66; 2 l-tiles at m = 0 with no parity split
        (257, 257),  # K and H > 128 with tails; 514 columns: a 512 + 2 split at programs=1
    ],
)
def test_kernels_match_torch(lmax, nfields, programs):
    from trnfft.sht import SHT

    _check(SHT(lmax, fft="torch"), nfields)


def test_polar_mask_skips_tiles(programs):
    """Masked-off tiles are never read: they hold NaN in the kernel's table,
    yet the result is finite and equals the torch loop on the zeroed table."""
    from trnfft.sht import SHT, TILE, encode_plan, m_offset, parity_rows

    s = SHT(300, fft="torch")
    L, H = s.lmax, s.H
    mask = s.tile_mask.copy()
    live = np.argwhere(mask)
    off = live[::2]  # every other live tile, including whole output tiles
    mask[tuple(off.T)] = False
    table = s.table.clone()
    poisoned = s.table.clone()
    for p, kt, h in off:
        m, q = divmod(int(p), 2)
        r0 = m_offset(L, m) + (parity_rows(L, m, 0) if q else 0)
        k = parity_rows(L, m, q)
        rows = slice(r0 + kt * TILE, r0 + min(k, (kt + 1) * TILE))
        cols = slice(h * TILE, min(H, (h + 1) * TILE))
        table[rows, cols] = 0.0
        poisoned[rows, cols] = float("nan")
    s.table = table
    s.tile_mask = mask
    s.plan = encode_plan(L, H, mask)
    _check(s, 300, table_for_kernel=poisoned)


def test_chunked_launches(programs, monkeypatch):
    """Several launches over m ranges give the same result as one."""
    from trnfft.nki.sht import _chunks
    from trnfft.sht import SHT

    monkeypatch.setenv("TRNFFT_SHT_TILES_PER_LAUNCH", "40")
    s = SHT(255, fft="torch")
    assert len(_chunks(s.plan)) > 3
    _check(s, 4)


@pytest.mark.parametrize("fft", ["torch", "trnfft"])
def test_sht_round_trip_on_kernels(fft):
    """Full transform with the latitude stage on the kernels (and, for
    ``fft="trnfft"``, the longitude FFTs on trnfft's NKI kernels too)."""
    import trnfft
    from trnfft import ComplexTensor
    from trnfft.sht import SHT, n_alm

    L = 31  # nlon = 64: power-of-two FFTs
    g = torch.Generator().manual_seed(3)
    re = torch.randn(4, n_alm(L), generator=g, dtype=torch.float64)
    im = torch.randn(4, n_alm(L), generator=g, dtype=torch.float64)
    im[:, : L + 1] = 0.0  # m = 0 coefficients of a real field are real
    ref = SHT(L, dtype=torch.float64, fft="torch", legendre="torch")
    f = ref.inverse(ComplexTensor(re, im))
    trnfft.set_backend("nki")
    try:
        s = SHT(L, legendre="nki", fft=fft)
        a = s.forward(f.float())
        f2 = s.inverse(a)
    finally:
        trnfft.set_backend("auto")
    num = ((a.real.double() - re) ** 2 + (a.imag.double() - im) ** 2).sum().sqrt()
    den = (re**2 + im**2).sum().sqrt()
    assert (num / den).item() < 1e-6
    assert ((f2.double() - f).norm() / f.norm()).item() < 1e-6
