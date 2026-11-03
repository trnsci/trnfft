"""NKI kernels for the latitude (associated Legendre) stage of ``trnfft.sht``.

Both kernels run the ``2(L + 1)`` per-m, per-parity problems of an SHT in one
launch (or a few, see ``_chunks``): a ragged grouped GEMM whose shapes are
static and known when the plan is built (trnsci/trnblas#98 is the general
form; this is the SHT's own instance of it).

Layouts (see ``trnfft/sht.py``):

* ``table``: ``(n_alm, H)`` fp32, the parity-packed Legendre table. Problem
  ``p = 2m + q`` owns rows ``m_offset(m) + r0 .. + K_p`` (``r0 = 0`` for
  ``q = 0``, ``K_e(m)`` for ``q = 1``), one row per degree ``l``, ``H``
  northern-hemisphere latitudes as columns.
* ``x`` / inverse output: ``(2(L + 1) H, 2F)``, problem ``p`` in rows
  ``p H .. p H + H`` (latitude-major), the ``2F`` columns are the real and
  imaginary parts of every field.
* ``alm`` / forward output: ``(n_alm, 2F)`` in the triangular m-major order;
  problem ``p``'s degrees are every other row from ``m_offset(m) + q``. Those
  stride-2 rows are DMA access patterns (``out[r : r + 2k - 1 : 2]``), not
  gathers.

Per problem, the table tiles are loaded into SBUF once and stay stationary
while every moving tile of up to 512 columns streams through the Tensor
Engine into fp32 PSUM, so the table crosses HBM once per transform however
large F is. The forward (analysis) kernel needs ``P_p^T`` as the stationary
operand: each tile is read with a plain ``nl.load`` and transposed on the
Tensor Engine (``nisa.nc_transpose``), once per problem, not per field tile
(no ``nl.load_transpose2d``: trnfft#97, aws-neuron-sdk#1452).

Polar tiles: the static mask in ``plan`` lists the 128 x 128 table tiles
(128 degrees by 128 latitudes) that are not all zero. Masked-off tiles are
neither loaded nor multiplied; an output tile with no live tile is stored as
zeros. The mask is part of the ``plan`` string, a kernel argument, so it is
in the NKI trace-cache key (aws-neuron-sdk#1454: values captured from a
closure or global are not).

Both physical cores of a logical core (LNC=2): the kernel is launched
gridless; with ``num_programs() == 2`` each program takes half of the ``2F``
columns (with 2F even, program 0 the real parts, program 1 the imaginary
parts when the split lands there), so both run identical shapes and masks and
each holds its own copy of the table tiles in its SBUF.

Tail tiles: ``K_p``, ``H`` and the column count need not be multiples of 128
/ 128 / 512; the last partial tile of each is traced at its own static size.
"""

from __future__ import annotations

import os

import torch

from .dispatch import HAS_NKI, _use_simulator

if HAS_NKI:
    import nki
    import nki.isa as nisa
    import nki.language as nl

SHT_TILE = 128  # degrees per tile and latitudes per tile
SHT_TILE_N = 512  # moving-tile columns (one fp32 PSUM bank)


def _decode_plan(plan):
    """``"L:H:m0:m1:bits,..."`` -> ``(L, H, m0, m1, bits)``; ``bits[p]`` is the
    ``kt``-major 0/1 tile mask of problem ``p = 2m + q`` (all ``m``)."""
    head, body = plan.rsplit(":", 1)
    L, H, m0, m1 = (int(v) for v in head.split(":"))
    return L, H, m0, m1, body.split(",")


def _m_offset(L, m):
    return m * (2 * L + 3 - m) // 2


def _k_rows(L, m, q):
    first = m + q
    return 0 if first > L else (L - first) // 2 + 1


def chunk_plan(plan: str, m0: int, m1: int) -> str:
    """The plan string restricted to ``m0 <= m < m1``."""
    L, H, _, _, body = plan.split(":", 4)
    return f"{L}:{H}:{m0}:{m1}:{body}"


if HAS_NKI:

    def _program_columns(n_cols):
        """``(c0, NP)``: this program's column window of the ``2F`` columns."""
        n_prog = nl.num_programs() if nl.program_ndim() > 0 else 1
        if n_cols % n_prog != 0:  # 2F is even, so this only guards n_prog > 2
            n_prog = 1
        pid = nl.program_id(0) if n_prog > 1 else 0
        NP = n_cols // n_prog
        return pid * NP, NP

    def _column_tiles(NP):
        TN = min(NP, SHT_TILE_N)
        return TN, NP // TN, NP % TN

    @nki.jit
    def sht_legendre_forward_kernel(table, x, plan):
        """Analysis latitude stage for ``m0 <= m < m1``: returns ``(R, 2F)`` with
        ``R = m_offset(m1) - m_offset(m0)``, the coefficient rows of those m
        (row ``i`` is global row ``m_offset(m0) + i``).

        ``table`` is the full ``(n_alm, H)`` table, ``x`` the full
        ``(2(L+1) H, 2F)`` folded longitude coefficients (quadrature weights
        already applied): ``a[rows(p)] = T_p @ x[p H : p H + H]``.
        """
        L, H, m0, m1, bits = _decode_plan(plan)
        n_cols = x.shape[1]
        base = _m_offset(L, m0)
        R = _m_offset(L, m1) - base
        out = nl.ndarray((R, n_cols), dtype=x.dtype, buffer=nl.shared_hbm)
        c0, NP = _program_columns(n_cols)
        TN, n_full, n_tail = _column_tiles(NP)
        ht_n = (H + SHT_TILE - 1) // SHT_TILE

        for m in range(m0, m1):
            o = _m_offset(L, m)
            ke = _k_rows(L, m, 0)
            for q in range(2):
                k = _k_rows(L, m, q)
                if k == 0:
                    continue
                p = 2 * m + q
                r0 = o + (ke if q else 0)
                kt_n = (k + SHT_TILE - 1) // SHT_TILE
                live = [[bits[p][kt * ht_n + h] == "1" for h in range(ht_n)] for kt in range(kt_n)]
                # P_p^T tiles, stationary in SBUF for every column tile below.
                tt = {}
                for kt in range(kt_n):
                    ks = min(SHT_TILE, k - kt * SHT_TILE)
                    for h in range(ht_n):
                        if not live[kt][h]:
                            continue
                        hs = min(SHT_TILE, H - h * SHT_TILE)
                        t = nl.load(
                            table[
                                r0 + kt * SHT_TILE : r0 + kt * SHT_TILE + ks,
                                h * SHT_TILE : h * SHT_TILE + hs,
                            ]
                        )
                        pt = nl.ndarray((hs, ks), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_transpose(dst=pt, data=t, engine=nisa.engine.tensor)
                        st = nl.ndarray((hs, ks), dtype=table.dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=st, src=pt)
                        tt[(kt, h)] = st
                used_h = [h for h in range(ht_n) if any(live[kt][h] for kt in range(kt_n))]

                def _ntile(n0, ns, p=p, k=k, kt_n=kt_n, live=live, tt=tt, used_h=used_h, o=o, q=q):
                    xs = {}
                    for h in used_h:
                        hs = min(SHT_TILE, H - h * SHT_TILE)
                        xs[h] = nl.load(
                            x[
                                p * H + h * SHT_TILE : p * H + h * SHT_TILE + hs,
                                c0 + n0 : c0 + n0 + ns,
                            ]
                        )
                    for kt in range(kt_n):
                        ks = min(SHT_TILE, k - kt * SHT_TILE)
                        hv = [h for h in range(ht_n) if live[kt][h]]
                        sb = nl.ndarray((ks, ns), dtype=nl.float32, buffer=nl.sbuf)
                        if hv:
                            ps = nl.zeros((ks, ns), dtype=nl.float32, buffer=nl.psum)
                            for h in hv:
                                nisa.nc_matmul(
                                    dst=ps, stationary=tt[(kt, h)], moving=xs[h], accumulate=True
                                )
                            nisa.tensor_copy(dst=sb, src=ps)
                        else:
                            nisa.memset(dst=sb, value=0.0)
                        row = o - base + q + 2 * kt * SHT_TILE
                        nl.store(out[row : row + 2 * ks - 1 : 2, c0 + n0 : c0 + n0 + ns], value=sb)

                for n in nl.affine_range(n_full):
                    _ntile(n * TN, TN)
                if n_tail:
                    _ntile(n_full * TN, n_tail)
        return out

    @nki.jit
    def sht_legendre_inverse_kernel(table, alm, plan):
        """Synthesis latitude stage for ``m0 <= m < m1``: returns
        ``(2 (m1 - m0) H, 2F)``, problem ``p``'s ``H`` latitude rows at
        ``(p - 2 m0) H``: ``y_p = T_p^T @ alm[rows(p)]``.

        ``table`` is the full ``(n_alm, H)`` table, ``alm`` the full
        ``(n_alm, 2F)`` coefficients.
        """
        L, H, m0, m1, bits = _decode_plan(plan)
        n_cols = alm.shape[1]
        out = nl.ndarray((2 * (m1 - m0) * H, n_cols), dtype=alm.dtype, buffer=nl.shared_hbm)
        c0, NP = _program_columns(n_cols)
        TN, n_full, n_tail = _column_tiles(NP)
        ht_n = (H + SHT_TILE - 1) // SHT_TILE

        for m in range(m0, m1):
            o = _m_offset(L, m)
            ke = _k_rows(L, m, 0)
            for q in range(2):
                k = _k_rows(L, m, q)
                p = 2 * m + q
                y0 = (p - 2 * m0) * H
                r0 = o + (ke if q else 0)
                kt_n = (k + SHT_TILE - 1) // SHT_TILE
                if k:
                    live = [
                        [bits[p][kt * ht_n + h] == "1" for h in range(ht_n)] for kt in range(kt_n)
                    ]
                else:  # m = L, odd parity: no degrees, the output rows are zero
                    live = []
                tiles = {}
                for kt in range(kt_n):
                    ks = min(SHT_TILE, k - kt * SHT_TILE)
                    for h in range(ht_n):
                        if live[kt][h]:
                            hs = min(SHT_TILE, H - h * SHT_TILE)
                            tiles[(kt, h)] = nl.load(
                                table[
                                    r0 + kt * SHT_TILE : r0 + kt * SHT_TILE + ks,
                                    h * SHT_TILE : h * SHT_TILE + hs,
                                ]
                            )
                used_k = [kt for kt in range(kt_n) if any(live[kt])]

                def _ntile(n0, ns, k=k, live=live, tiles=tiles, used_k=used_k, o=o, q=q, y0=y0):
                    a_t = {}
                    for kt in used_k:
                        ks = min(SHT_TILE, k - kt * SHT_TILE)
                        row = o + q + 2 * kt * SHT_TILE
                        a_t[kt] = nl.load(alm[row : row + 2 * ks - 1 : 2, c0 + n0 : c0 + n0 + ns])
                    for h in range(ht_n):
                        hs = min(SHT_TILE, H - h * SHT_TILE)
                        kv = [kt for kt in used_k if live[kt][h]]
                        sb = nl.ndarray((hs, ns), dtype=nl.float32, buffer=nl.sbuf)
                        if kv:
                            ps = nl.zeros((hs, ns), dtype=nl.float32, buffer=nl.psum)
                            for kt in kv:
                                nisa.nc_matmul(
                                    dst=ps,
                                    stationary=tiles[(kt, h)],
                                    moving=a_t[kt],
                                    accumulate=True,
                                )
                            nisa.tensor_copy(dst=sb, src=ps)
                        else:
                            nisa.memset(dst=sb, value=0.0)
                        r = y0 + h * SHT_TILE
                        nl.store(out[r : r + hs, c0 + n0 : c0 + n0 + ns], value=sb)

                for n in nl.affine_range(n_full):
                    _ntile(n * TN, TN)
                if n_tail:
                    _ntile(n_full * TN, n_tail)
        return out

else:  # pragma: no cover - CPU-only installs
    sht_legendre_forward_kernel = None
    sht_legendre_inverse_kernel = None


# ── Host launchers ───────────────────────────────────────────────────────────


def _tiles_per_m(plan: str) -> list[int]:
    L, H, _, _, bits = _decode_plan(plan)
    return [bits[2 * m].count("1") + bits[2 * m + 1].count("1") for m in range(L + 1)]


def _chunks(plan: str) -> list[tuple[int, int]]:
    """Split ``0 <= m <= L`` into contiguous launches of at most
    ``TRNFFT_SHT_TILES_PER_LAUNCH`` live table tiles each (default 8192; 0 =
    one launch). Each launch is a separate static trace, so this bounds the
    traced program size; tiles per launch, not m per launch, because the
    per-m work falls off as (L - m)."""
    L = int(plan.split(":", 1)[0])
    cap = int(os.environ.get("TRNFFT_SHT_TILES_PER_LAUNCH", "8192") or 0)
    if cap <= 0:
        return [(0, L + 1)]
    out, start, acc = [], 0, 0
    for m, t in enumerate(_tiles_per_m(plan)):
        if acc and acc + t > cap:
            out.append((start, m))
            start, acc = m, 0
        acc += t
    out.append((start, L + 1))
    return out


def _run(kernel, op_name: str, table: torch.Tensor, data: torch.Tensor, plan: str):
    """One launch: simulator, native TorchNeuron, or XLA."""
    from . import dispatch as _d

    # Structural dtype guard (trnsci#35), on every route (``_launch`` checks
    # again): the kernels take float32 only.
    _d._check_kernel_dtypes(op_name, (table, data))
    if _use_simulator():
        import numpy as np

        programs = int(os.environ.get("TRNFFT_SHT_SIM_PROGRAMS", "1"))
        k = kernel[programs] if programs > 1 else kernel
        out = nki.simulate(k)(table.detach().cpu().numpy(), data.detach().cpu().numpy(), plan=plan)
        return torch.from_numpy(np.asarray(out)).to(data.device)
    if _d._is_native() and os.environ.get("TRNFFT_SHT_LAUNCH", "op").strip() == "direct":
        dev = _d._NEURON_DEVICE
        t, x = table.to(dev), data.to(dev)
        with _d._LAUNCH_LOCK:
            out = kernel(t, x, plan=plan)
        return out if data.device.type == "neuron" else _d._to_host(out, data.device)
    return _d._launch(kernel, op_name, table, data, plan)


def legendre_forward_nki(table, x, lmax: int, H: int, plan: str) -> torch.Tensor:
    """Analysis latitude stage on the NKI kernel: ``(n_alm, 2F)``."""
    if not HAS_NKI:
        raise RuntimeError("the NKI SHT kernels need nki")
    x = x.contiguous()
    parts = [
        _run(sht_legendre_forward_kernel, "sht_legendre_forward", table, x, chunk_plan(plan, a, b))
        for a, b in _chunks(plan)
    ]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def legendre_inverse_nki(table, alm, lmax: int, H: int, plan: str) -> torch.Tensor:
    """Synthesis latitude stage on the NKI kernel: ``(2 (L+1) H, 2F)``."""
    if not HAS_NKI:
        raise RuntimeError("the NKI SHT kernels need nki")
    alm = alm.contiguous()
    parts = [
        _run(
            sht_legendre_inverse_kernel, "sht_legendre_inverse", table, alm, chunk_plan(plan, a, b)
        )
        for a, b in _chunks(plan)
    ]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
