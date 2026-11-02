"""Spherical harmonic transforms on Trainium (trnfft#148, trnsci/trnsci#111).

A scalar spherical harmonic transform (SHT) of a real field ``f(theta, phi)``
on a latitude-longitude grid, batched over ``F`` fields:

    f(theta, phi) = sum_{l=0..L} sum_{m=-l..l} a_lm Y_lm(theta, phi)

The transform has two stages:

1. **Longitude.** A real FFT along each latitude ring (``trnfft.rfft`` /
   ``trnfft.irfft``) gives the zonal Fourier coefficients ``G_m(theta)``.
2. **Latitude.** For every zonal wavenumber ``m`` a dense GEMM against a table
   of the normalised associated Legendre functions ``P_lm(cos theta)``:
   ``a_m = P_m (w * G_m)`` (analysis) and ``G_m = P_m^T a_m`` (synthesis).

The latitude stage is O(L^3) per field and dominates at high L. On Trainium it
is one NKI kernel launch (``trnfft/nki/sht.py``) in which each ``P_m`` is held
stationary in SBUF while all ``2F`` real columns (real and imaginary parts of
every field) stream through the Tensor Engine into fp32 PSUM, so the table is
read from HBM once per transform rather than once per field.

Conventions
-----------
* Orthonormal harmonics, ``integral |Y_lm|^2 dOmega = 1``, with the
  Condon-Shortley phase ``(-1)^m``. Same as ``ducc0.sht`` and SHTns with
  ``SHT_ORTHONORMAL``; checked against both in ``tests/test_sht.py``.
* Coefficients are complex, ``m >= 0`` only (the field is real, so
  ``a_{l,-m} = (-1)^m conj(a_lm)``), in the triangular m-major order used by
  ducc0 and healpy: ``index(l, m) = m * (2L + 3 - m) / 2 + (l - m)``.
  ``alm`` has shape ``(..., (L + 1)(L + 2) / 2)`` as a :class:`ComplexTensor`.
* Grids, latitude index 0 at the north pole:
    - ``"gauss"``: Gauss-Legendre nodes, ``nlat = L + 1`` by default (exact
      quadrature for band-limited fields);
    - ``"equiangular"``: ``theta_j = pi j / (nlat - 1)`` including both poles,
      Clenshaw-Curtis weights, ``nlat = 2L + 2`` by default (exact for
      band-limited fields from ``nlat >= 2L + 1``).
  ``nlon = 2L + 2`` by default (power of two for L = 511, 1023).

Even/odd symmetry
-----------------
``P_lm(-x) = (-1)^(l+m) P_lm(x)``, so each ``P_m`` splits into an even and an
odd part over the northern half of the grid (``H = ceil(nlat / 2)`` rows), and
each of the ``2(L + 1)`` sub-problems ``p = 2m + parity`` is a GEMM of shape
``(K_p x H)`` with ``K_p ~ (L - m + 1) / 2``. That halves the flops and the
table. Within one ``m`` block of the coefficient layout the rows of parity
``q`` are ``l = m + q, m + q + 2, ...``: a stride-2 row access, which the NKI
kernels express as strided DMA access patterns, not gathers.

Tables
------
Computed once on the host in fp64 by the standard three-term recurrence in
``l``, with ``P_mm`` started in log space and a running power-of-two exponent
(so ``sin(theta)^m`` at ``m ~ 1000`` near the poles does not underflow fp64
before the recurrence has grown it back), then rounded once to fp32. Entries
below ``skip_threshold`` (default 1e-30) times the largest ``|P_lm|`` of
their ``m`` are set to zero. A 128 x 128 tile (128 values of ``l`` by 128
latitudes) that is all zero after that is a *polar tile*: the static mask
:attr:`SHT.tile_mask` lists them and the kernels skip them. See
``docs/api/sht.md`` for the accuracy analysis.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from .complex import ComplexTensor

TILE = 128  # Tensor Engine partition tile, for both l and latitude

__all__ = [
    "SHT",
    "gauss_legendre_grid",
    "equiangular_grid",
    "legendre_table",
    "lm_index",
    "n_alm",
]


# ── Index helpers ────────────────────────────────────────────────────────────


def n_alm(lmax: int) -> int:
    """Number of complex coefficients ``a_lm`` with ``0 <= m <= l <= lmax``."""
    return (lmax + 1) * (lmax + 2) // 2


def m_offset(lmax: int, m: int) -> int:
    """Row of ``a_mm`` in the triangular m-major layout."""
    return m * (2 * lmax + 3 - m) // 2


def lm_index(lmax: int, l: int, m: int) -> int:
    """Index of ``a_lm`` in the triangular m-major layout (ducc0 / healpy order)."""
    if not 0 <= m <= l <= lmax:
        raise ValueError(f"need 0 <= m <= l <= lmax, got l={l}, m={m}, lmax={lmax}")
    return m_offset(lmax, m) + l - m


def parity_rows(lmax: int, m: int, parity: int) -> int:
    """``K_p``: number of degrees ``l = m + parity, m + parity + 2, ... <= lmax``."""
    first = m + parity
    return 0 if first > lmax else (lmax - first) // 2 + 1


# ── Grids ────────────────────────────────────────────────────────────────────


def gauss_legendre_grid(nlat: int) -> tuple[np.ndarray, np.ndarray]:
    """``(x, w)``: Gauss-Legendre nodes ``x = cos(theta)`` (north to south) and
    weights on [-1, 1], in fp64. Exact for polynomials of degree ``2 nlat - 1``.

    Nodes are polished with Newton steps on the three-term recurrence, so they
    are accurate to a few ulps at nlat ~ 2000 (``numpy.polynomial.legendre.
    leggauss`` alone loses a few digits there).
    """
    x, _ = np.polynomial.legendre.leggauss(nlat)
    x = x.astype(np.float64)
    for _ in range(3):
        p0 = np.ones_like(x)
        p1 = x.copy()
        for k in range(2, nlat + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        # P_n = p1, P_{n-1} = p0; P_n' = n (x P_n - P_{n-1}) / (x^2 - 1)
        dp = nlat * (x * p1 - p0) / (x * x - 1.0)
        x = x - p1 / dp
    p0 = np.ones_like(x)
    p1 = x.copy()
    for k in range(2, nlat + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    dp = nlat * (x * p1 - p0) / (x * x - 1.0)
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    return x[::-1].copy(), w[::-1].copy()


def _clenshaw_curtis_weights(n: int) -> np.ndarray:
    """Clenshaw-Curtis weights for the ``n + 1`` points ``cos(pi j / n)``
    (Waldvogel 2006, the DFT construction), fp64. Exact for polynomials of
    degree ``n``."""
    if n == 1:
        return np.array([1.0, 1.0])
    odd = np.arange(1, n, 2)
    lo = len(odd)
    mm = n - lo
    v0 = np.concatenate([2.0 / odd / (odd - 2.0), [1.0 / odd[-1]], np.zeros(mm)])
    v2 = -v0[:-1] - v0[:0:-1]
    g0 = -np.ones(n)
    g0[lo] += n
    g0[mm] += n
    g = g0 / (n * n - 1 + (n % 2))
    w = np.fft.ifft(v2 + g).real
    return np.concatenate([w, w[:1]])


def equiangular_grid(nlat: int) -> tuple[np.ndarray, np.ndarray]:
    """``(x, w)``: equiangular nodes ``cos(pi j / (nlat - 1))`` including both
    poles (north to south) and Clenshaw-Curtis weights on [-1, 1], fp64."""
    if nlat < 2:
        raise ValueError("equiangular grid needs nlat >= 2")
    n = nlat - 1
    x = np.cos(np.pi * np.arange(nlat) / n)
    return x, _clenshaw_curtis_weights(n)


# ── Legendre tables (host, fp64) ─────────────────────────────────────────────


def legendre_table(lmax: int, x: np.ndarray) -> np.ndarray:
    """Normalised associated Legendre functions in the parity-packed layout, fp64.

    Returns ``T`` of shape ``(n_alm(lmax), len(x))``. Rows ``m_offset(m) ..
    m_offset(m) + K_e`` hold ``l = m, m + 2, ...`` and the next ``K_o`` rows
    hold ``l = m + 1, m + 3, ...``. ``T[row(l, m), j] = Pbar_lm(x_j) /
    sqrt(2 pi)`` with ``Y_lm = T e^{i m phi}`` orthonormal on the sphere
    (Condon-Shortley phase included).

    The recurrence runs over ``d = l - m`` with all ``m`` at once. ``P_mm`` is
    formed in log space and carried as an fp64 mantissa times ``2^e``, rescaled
    whenever the mantissa grows past 2^256, so no intermediate underflows.
    """
    x = np.asarray(x, dtype=np.float64)
    L = lmax
    M = L + 1
    J = x.size
    sin_t = np.sqrt(np.maximum(0.0, (1.0 - x) * (1.0 + x)))
    with np.errstate(divide="ignore"):
        log_sin = np.log(sin_t)  # -inf at the poles
    m = np.arange(M, dtype=np.float64)
    # log |Pbar_mm / sqrt(2 pi)| = log(1/sqrt(4 pi)) + sum_k 0.5 log((2k+1)/(2k)) + m log sin
    k = np.arange(1, M, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * np.log((2 * k + 1) / (2 * k)))])
    with np.errstate(invalid="ignore"):  # 0 * -inf at the poles for m = 0, reset below
        log_pmm = (-0.5 * math.log(4 * math.pi) + cum)[:, None] + m[:, None] * log_sin[None, :]
    log_pmm[0, :] = -0.5 * math.log(4 * math.pi)  # m = 0: no sin factor (also at poles)
    ln2 = math.log(2.0)
    finite = np.isfinite(log_pmm)
    e = np.where(finite, np.floor(np.where(finite, log_pmm, 0.0) / ln2), 0.0)
    mant = np.where(finite, np.exp(np.where(finite, log_pmm, 0.0) - e * ln2), 0.0)
    sign = np.where(np.arange(M) % 2 == 1, -1.0, 1.0)[:, None]  # Condon-Shortley
    cur = sign * mant  # (M, J): Pbar_{m+d, m} mantissas, d = 0
    prev = np.zeros_like(cur)
    e = e.astype(np.int64)

    T = np.zeros((n_alm(L), J), dtype=np.float64)
    offs = np.array([m_offset(L, mm) for mm in range(M)], dtype=np.int64)
    k_even = np.array([parity_rows(L, mm, 0) for mm in range(M)], dtype=np.int64)

    def _write(d: int, vals: np.ndarray, exps: np.ndarray):
        ms = np.arange(M - d)
        rows = offs[ms] + np.where(d % 2 == 0, d // 2, k_even[ms] + d // 2)
        T[rows] = np.ldexp(vals, exps)

    _write(0, cur, e)
    big = 2.0**256
    for d in range(1, M):
        n_act = M - d
        ms = m[:n_act]
        l = ms + d
        a = np.sqrt((4 * l * l - 1) / (l * l - ms * ms))[:, None]
        if d == 1:
            new = a * x[None, :] * cur[:n_act]
        else:
            lp = l - 1
            a_prev = np.sqrt((4 * lp * lp - 1) / (lp * lp - ms * ms))[:, None]
            new = a * (x[None, :] * cur[:n_act] - prev[:n_act] / a_prev)
        prev = cur[:n_act]
        cur = new
        e = e[:n_act]
        hot = np.abs(cur) > big
        if hot.any():
            cur = np.where(hot, cur / big, cur)
            prev = np.where(hot, prev / big, prev)
            e = e + np.where(hot, 256, 0)
        _write(d, cur, e)
    return T


# ── Plan ─────────────────────────────────────────────────────────────────────


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


class SHT:
    """Scalar spherical harmonic transform plan (forward = analysis, inverse = synthesis).

    Args:
        lmax: band limit ``L`` (``mmax = L``).
        nlat, nlon: grid size; defaults ``nlat = L + 1`` (gauss) or ``2L + 2``
            (equiangular), ``nlon = 2L + 2``.
        grid: ``"gauss"`` or ``"equiangular"``.
        dtype: table and compute dtype, ``torch.float32`` (default, the device
            dtype) or ``torch.float64`` (CPU reference).
        skip_threshold: table entries below this times the largest ``|P_lm|``
            of their ``m`` are dropped; all-zero 128 x 128 tiles are skipped.
            ``0`` keeps every entry (no polar skipping beyond exact zeros).
        legendre: ``"auto"`` (NKI kernel when the trnfft backend selects NKI,
            else torch), ``"torch"`` (per-m aten GEMM loop), or ``"nki"``.
        fft: ``"trnfft"`` (default: ``trnfft.rfft`` / ``trnfft.irfft``, NKI
            on device) or ``"torch"`` (``torch.fft``, for CPU baselines).
        device: device the tables live on (uploaded once).

    The tables are built at construction (a few seconds at L = 1023) and kept
    on ``device``. ``forward(f)`` takes ``(..., nlat, nlon)`` real fields and
    returns a :class:`ComplexTensor` of shape ``(..., n_alm)``; ``inverse``
    is the reverse. Leading dimensions are flattened into the field batch F.
    """

    def __init__(
        self,
        lmax: int,
        nlat: int | None = None,
        nlon: int | None = None,
        grid: str = "gauss",
        dtype: torch.dtype = torch.float32,
        skip_threshold: float = 1e-30,
        legendre: str = "auto",
        fft: str = "trnfft",
        device: torch.device | str | None = None,
    ):
        if lmax < 0:
            raise ValueError("lmax must be >= 0")
        if grid not in ("gauss", "equiangular"):
            raise ValueError(f"grid must be 'gauss' or 'equiangular', got {grid!r}")
        if legendre not in ("auto", "torch", "nki"):
            raise ValueError(f"legendre must be 'auto', 'torch' or 'nki', got {legendre!r}")
        if fft not in ("trnfft", "torch"):
            raise ValueError(f"fft must be 'trnfft' or 'torch', got {fft!r}")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        L = lmax
        self.lmax = L
        self.grid = grid
        self.nlat = nlat if nlat is not None else (L + 1 if grid == "gauss" else 2 * L + 2)
        self.nlon = nlon if nlon is not None else 2 * L + 2
        if grid == "gauss" and self.nlat < L + 1:
            raise ValueError(f"gauss grid needs nlat >= lmax + 1 = {L + 1}")
        if grid == "equiangular" and self.nlat < 2 * L + 1:
            raise ValueError(f"equiangular grid needs nlat >= 2 lmax + 1 = {2 * L + 1}")
        if self.nlon < 2 * L + 1:
            raise ValueError(f"nlon must be >= 2 lmax + 1 = {2 * L + 1}")
        self.dtype = dtype
        self.skip_threshold = float(skip_threshold)
        self.legendre = legendre
        self.fft = fft
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        x, w = (gauss_legendre_grid if grid == "gauss" else equiangular_grid)(self.nlat)
        self.x = x  # cos(theta), fp64, north to south
        self.theta = np.arccos(np.clip(x, -1.0, 1.0))
        self.weights = w  # fp64, sum 2
        H = (self.nlat + 1) // 2
        self.H = H
        self.n_alm = n_alm(L)

        T64 = legendre_table(L, x[:H])
        mask, T64 = self._polar_mask(T64)
        self.tile_mask = mask  # (2(L+1), kt_max, ht) bool: True = tile computed
        np_dtype = np.float32 if dtype == torch.float32 else np.float64
        table = T64.astype(np_dtype)
        self.table_rounding_error = float(
            np.abs(table.astype(np.float64) - T64).max() / max(np.abs(T64).max(), 1e-300)
        )
        self.table = torch.from_numpy(table).to(self.device)  # (n_alm, H)
        # Analysis weights: quadrature x 2 pi / nlon (the rfft sum -> integral
        # over phi). For odd nlat the equator row is its own mirror, so it is
        # halved before the north + south fold counts it twice.
        wq = w[:H].copy() * (2.0 * math.pi / self.nlon)
        if self.nlat % 2 == 1:
            wq[H - 1] *= 0.5
        self._wq = torch.from_numpy(wq).to(dtype=dtype, device=self.device)
        self.plan = encode_plan(L, H, mask)

    # ── static mask ──────────────────────────────────────────────────────

    def _polar_mask(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        L, H = self.lmax, T.shape[1]
        M = L + 1
        ht = -(-H // TILE)
        kt_max = -(-parity_rows(L, 0, 0) // TILE)
        mask = np.zeros((2 * M, kt_max, ht), dtype=bool)
        for m in range(M):
            o = m_offset(L, m)
            blk = T[o : o + L - m + 1]
            if self.skip_threshold > 0:
                cut = self.skip_threshold * np.abs(blk).max()
                blk[np.abs(blk) < cut] = 0.0
            ke = parity_rows(L, m, 0)
            for par, (r0, k) in enumerate(((0, ke), (ke, parity_rows(L, m, 1)))):
                sub = blk[r0 : r0 + k]
                for kt in range(-(-k // TILE)):
                    for h in range(ht):
                        t = sub[kt * TILE : (kt + 1) * TILE, h * TILE : (h + 1) * TILE]
                        mask[2 * m + par, kt, h] = bool(np.any(t != 0.0))
        return mask, T

    @property
    def tile_counts(self) -> tuple[int, int]:
        """``(computed, total)`` number of 128 x 128 table tiles."""
        L, H = self.lmax, self.H
        ht = -(-H // TILE)
        total = 0
        for m in range(L + 1):
            for par in (0, 1):
                total += -(-parity_rows(L, m, par) // TILE) * ht
        return int(self.tile_mask.sum()), total

    def flops(self, nfields: int) -> int:
        """Real flops of one latitude stage (either direction) at ``nfields``,
        counting only the tiles that are computed: ``2 * K * H * 2F`` per
        problem over the unmasked rows/columns."""
        L, H = self.lmax, self.H
        total = 0
        for m in range(L + 1):
            for par in (0, 1):
                k = parity_rows(L, m, par)
                for kt in range(-(-k // TILE)):
                    ks = min(TILE, k - kt * TILE)
                    for h in range(-(-H // TILE)):
                        if self.tile_mask[2 * m + par, kt, h]:
                            total += 2 * ks * min(TILE, H - h * TILE) * 2 * nfields
        return total

    # ── device placement ─────────────────────────────────────────────────

    def to(self, device) -> SHT:
        """Move the tables to ``device`` (once; transforms then stay there)."""
        self.device = torch.device(device)
        self.table = self.table.to(self.device)
        self._wq = self._wq.to(self.device)
        return self

    # ── transforms ───────────────────────────────────────────────────────

    def forward(self, f: torch.Tensor) -> ComplexTensor:
        """Analysis: real fields ``(..., nlat, nlon)`` -> ``a_lm`` ``(..., n_alm)``."""
        lead = f.shape[:-2]
        if tuple(f.shape[-2:]) != (self.nlat, self.nlon):
            raise ValueError(
                f"expected (..., {self.nlat}, {self.nlon}) fields, got {tuple(f.shape)}"
            )
        F = math.prod(lead) if lead else 1
        x = self.longitude_forward(f.reshape(F, self.nlat, self.nlon).to(self.dtype))
        y = self.legendre_forward(x)
        return self.unpack_alm(y, lead)

    def inverse(self, alm: ComplexTensor) -> torch.Tensor:
        """Synthesis: ``a_lm`` ``(..., n_alm)`` -> real fields ``(..., nlat, nlon)``."""
        if not isinstance(alm, ComplexTensor):
            alm = ComplexTensor(alm.real, alm.imag) if torch.is_complex(alm) else alm
        if alm.real.shape[-1] != self.n_alm:
            raise ValueError(f"expected (..., {self.n_alm}) coefficients, got {alm.real.shape}")
        lead = alm.real.shape[:-1]
        a = self.pack_alm(alm)
        g = self.legendre_inverse(a)
        return self.longitude_inverse(g, lead)

    __call__ = forward

    # ── stages (public so the benchmark can time them separately) ───────

    def _rfft(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fft == "torch":
            z = torch.fft.rfft(f, dim=-1)
            return z.real, z.imag
        from .api import rfft

        z = rfft(f, dim=-1)
        return z.real, z.imag

    def _irfft(self, re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
        if self.fft == "torch":
            return torch.fft.irfft(torch.complex(re, im), n=self.nlon, dim=-1, norm="forward")
        from .api import irfft

        return irfft(ComplexTensor(re, im), n=self.nlon, dim=-1, norm="forward")

    def longitude_forward(self, f: torch.Tensor) -> torch.Tensor:
        """``(F, nlat, nlon)`` real -> the packed GEMM operand ``X`` of shape
        ``(2(L+1) H, 2F)``: rows ``p H .. p H + H`` are problem ``p = 2m + q``
        (``q`` = 0: north + south fold, 1: north - south), columns are
        ``[Re over fields | Im over fields]``.

        FFT, weights, the even/odd fold, then one permute to m-major."""
        F = f.shape[0]
        M, H = self.lmax + 1, self.H
        re, im = self._rfft(f)  # (F, nlat, nlon//2+1)
        g = torch.stack((re[..., :M], im[..., :M]), dim=-1)  # (F, nlat, M, 2)
        north = g[:, :H]
        south = torch.flip(g, (1,))[:, :H]  # row i: latitude nlat - 1 - i
        wq = self._wq.to(device=g.device, dtype=g.dtype).view(1, H, 1, 1)
        north = north * wq
        south = south * wq
        fold = torch.stack((north + south, north - south), dim=3)  # (F, H, M, 2q, 2ri)
        # -> (M, q, H, ri, F)
        return fold.permute(2, 3, 1, 4, 0).reshape(2 * M * H, 2 * F).contiguous()

    def longitude_inverse(self, g: torch.Tensor, lead=None) -> torch.Tensor:
        """Packed GEMM output ``(2(L+1) H, 2F)`` -> real fields ``(F..., nlat, nlon)``."""
        M, H, nlat = self.lmax + 1, self.H, self.nlat
        F = g.shape[1] // 2
        y = g.reshape(M, 2, H, 2, F)
        s, a = y[:, 0], y[:, 1]
        north = s + a  # (M, H, ri, F)
        south = torch.flip((s - a)[:, : nlat // 2], (1,))
        grid = torch.cat((north, south), dim=1)  # (M, nlat, ri, F)
        grid = grid.permute(2, 3, 1, 0)  # (ri, F, nlat, M)
        nf = self.nlon // 2 + 1
        pad = (0, nf - M)
        re = torch.nn.functional.pad(grid[0], pad).contiguous()
        im = torch.nn.functional.pad(grid[1], pad).contiguous()
        out = self._irfft(re, im)
        if lead is not None:
            out = out.reshape(*lead, nlat, self.nlon)
        return out

    def unpack_alm(self, y: torch.Tensor, lead=None) -> ComplexTensor:
        """``(n_alm, 2F)`` -> ComplexTensor ``(..., n_alm)`` (``lead`` defaults to ``(F,)``)."""
        F = y.shape[1] // 2
        if lead is None:
            lead = (F,)
        z = y.reshape(self.n_alm, 2, F).permute(1, 2, 0)
        re = z[0].reshape(*lead, self.n_alm).contiguous()
        im = z[1].reshape(*lead, self.n_alm).contiguous()
        return ComplexTensor(re, im)

    def pack_alm(self, alm: ComplexTensor) -> torch.Tensor:
        """ComplexTensor ``(..., n_alm)`` -> ``(n_alm, 2F)`` (fields as columns)."""
        re = alm.real.reshape(-1, self.n_alm).to(self.dtype)
        im = alm.imag.reshape(-1, self.n_alm).to(self.dtype)
        return torch.stack((re, im), dim=0).permute(2, 0, 1).reshape(self.n_alm, -1).contiguous()

    def _use_nki_legendre(self) -> bool:
        if self.legendre == "torch":
            return False
        if self.legendre == "nki":
            return True
        from .nki.dispatch import _use_nki

        return _use_nki() and self.dtype == torch.float32

    def legendre_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Latitude analysis: ``X (2(L+1) H, 2F)`` -> ``a (n_alm, 2F)``."""
        if self._use_nki_legendre():
            from .nki.sht import legendre_forward_nki

            return legendre_forward_nki(self.table, x, self.lmax, self.H, self.plan)
        return legendre_forward_torch(self.table, x, self.lmax, self.H)

    def legendre_inverse(self, a: torch.Tensor) -> torch.Tensor:
        """Latitude synthesis: ``a (n_alm, 2F)`` -> ``Y (2(L+1) H, 2F)``."""
        if self._use_nki_legendre():
            from .nki.sht import legendre_inverse_nki

            return legendre_inverse_nki(self.table, a, self.lmax, self.H, self.plan)
        return legendre_inverse_torch(self.table, a, self.lmax, self.H)


# ── PyTorch latitude stage (the per-m aten GEMM loop) ───────────────────────


def legendre_forward_torch(table: torch.Tensor, x: torch.Tensor, lmax: int, H: int):
    """``a[rows(m, q)] = T_p @ X_p`` for every problem ``p = 2m + q``: one
    matmul per problem (the per-m aten loop of trnsci/trnsci#111)."""
    L = lmax
    out = x.new_empty((n_alm(L), x.shape[1]))
    for m in range(L + 1):
        o = m_offset(L, m)
        ke = parity_rows(L, m, 0)
        for q, (r0, k) in enumerate(((0, ke), (ke, parity_rows(L, m, 1)))):
            if k == 0:
                continue
            p = 2 * m + q
            t = table[o + r0 : o + r0 + k]  # (k, H)
            out[o + q : o + q + 2 * k - 1 : 2] = t @ x[p * H : (p + 1) * H]
    return out


def legendre_inverse_torch(table: torch.Tensor, a: torch.Tensor, lmax: int, H: int):
    """``Y_p = T_p^T @ a[rows(m, q)]`` for every problem ``p = 2m + q``."""
    L = lmax
    out = a.new_zeros((2 * (L + 1) * H, a.shape[1]))
    for m in range(L + 1):
        o = m_offset(L, m)
        ke = parity_rows(L, m, 0)
        for q, (r0, k) in enumerate(((0, ke), (ke, parity_rows(L, m, 1)))):
            if k == 0:
                continue
            p = 2 * m + q
            t = table[o + r0 : o + r0 + k]
            out[p * H : (p + 1) * H] = t.T @ a[o + q : o + q + 2 * k - 1 : 2]
    return out


# ── Static plan string (kernel argument) ─────────────────────────────────────


def encode_plan(lmax: int, H: int, mask: np.ndarray) -> str:
    """Encode ``(lmax, H, tile mask)`` as the static string the NKI kernels take.

    The mask is a kernel *argument*, part of the NKI trace-cache key, never a
    closure or global (aws-neuron-sdk#1454). Format: ``"L:H:m0:m1:"`` (the
    launch covers ``m0 <= m < m1``; the full plan has ``0:L+1``) followed by,
    for each problem ``p = 2m + q`` with ``K_p > 0``, its ``kt * ht`` tile bits
    (``kt``-major) as ``0``/``1``, problems separated by ``,``.
    """
    L = lmax
    ht = -(-H // TILE)
    parts = []
    for m in range(L + 1):
        for q in (0, 1):
            k = parity_rows(L, m, q)
            kt = -(-k // TILE)
            bits = mask[2 * m + q, :kt, :ht].reshape(-1)
            parts.append("".join("1" if b else "0" for b in bits))
    return f"{L}:{H}:0:{L + 1}:" + ",".join(parts)
