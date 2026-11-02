"""
NKI dispatch layer — auto-selects between PyTorch and NKI backends.

Folded from neuron-complex-ops. When running on Trainium with neuronxcc
available, dispatches to optimized NKI kernels. Otherwise uses PyTorch ops.

Backend selection:
    "auto"    — NKI if available, else PyTorch
    "pytorch" — Always PyTorch (works on CPU, CUDA, Trainium via TorchNeuron)
    "nki"     — Always NKI (fails if neuronxcc not installed)
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import threading
import warnings

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:
    HAS_NKI = False

import numpy as np
import torch

from ..complex import ComplexTensor, complex_matmul

PMAX = 128  # Max partition dimension (systolic array rows)
# SBUF bytes per partition the complex linear kernel may spend on keeping
# W^T resident for the whole launch (``_complex_gemm_tiles``): three planes of
# ceil(in/128) * out elements. 64 KiB of the 192 KiB: FP32 W of (512, 1024) or
# (1024, 512) fits, (1024, 1024) keeps the per-tile transposes.
_HOIST_SBUF_BYTES = 64 * 1024

# When set, dispatch bypasses torch_xla and runs kernels through
# `nki.simulate(kernel)(np_args)` on CPU. Lets us iterate kernels on any
# x86_64 Linux box without paying the NEFF compile + hardware dispatch
# cost. Semantics follow NKI 0.3.0's simulator: no NEFF compile, no
# SBUF/PSUM capacity checks, no latency/parallelism modelling. For
# correctness iteration only; hardware still owns perf numbers.
_USE_SIMULATOR = os.environ.get("TRNFFT_USE_SIMULATOR", "").lower() in (
    "1",
    "true",
    "yes",
)


def _use_simulator() -> bool:
    return _USE_SIMULATOR and HAS_NKI


_backend = "auto"


def set_backend(backend: str):
    """Set dispatch backend: 'auto', 'pytorch', or 'nki'."""
    global _backend
    assert backend in ("auto", "pytorch", "nki")
    if backend == "nki" and not HAS_NKI:
        raise RuntimeError("NKI backend requires nki>=0.3.0 (Neuron SDK 2.29+)")
    _backend = backend


def get_backend() -> str:
    return _backend


def _use_nki() -> bool:
    if _backend == "nki":
        return True
    if _backend == "pytorch":
        return False
    return HAS_NKI


def complex_gemm(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """Complex matrix multiply with backend dispatch.

    NKI kernel uses stationary tile reuse:
    - Phase 1: A_real stationary, stream B_real and B_imag
    - Phase 2: A_imag stationary, stream -B_imag and B_real
    - 4 SBUF loads instead of 8, PSUM accumulation
    """
    if _use_nki():
        return _nki_complex_gemm(a, b)
    return complex_matmul(a, b)


# complex_gemm_bf16 is defined later in the file (after the NKI kernels),
# because it references _complex_gemm_kernel_bf16 which only exists when HAS_NKI.


def complex_linear(x: ComplexTensor, w_real: torch.Tensor, w_imag: torch.Tensor) -> ComplexTensor:
    """Complex linear forward: y = x @ W^T (complex) with backend dispatch.

    NKI kernel fuses the 4 real matmuls and reuses the activation tile
    (loaded once, streamed against W_real and W_imag).
    """
    if _use_nki():
        return _nki_complex_linear(x, w_real, w_imag)
    # PyTorch fallback: 4 real matmuls
    yr = torch.matmul(x.real, w_real.t()) - torch.matmul(x.imag, w_imag.t())
    yi = torch.matmul(x.real, w_imag.t()) + torch.matmul(x.imag, w_real.t())
    return ComplexTensor(yr, yi)


def complex_mask_apply(mask: ComplexTensor, spec: ComplexTensor) -> ComplexTensor:
    """Apply complex mask to spectrogram with backend dispatch.

    NKI kernel fuses all 6 element-wise ops into single kernel invocation,
    avoiding 6 HBM round-trips.
    """
    if _use_nki():
        return _nki_complex_mask(mask, spec)
    return mask * spec


# Tensor-engine transposes instead of ``nl.load_transpose2d`` (trnfft#97).
# On Trn2 (SDK 2.33, native TorchNeuron) the runtime rebuilds a kernel's IO
# DMA rings for a ``load_transpose2d`` input every time that input's address
# changes, which is every call with a fresh input tensor. The DFT-GEMM
# (``_complex_gemm_kernel``, the default FFT for N < 512), the radix-8
# Stockham W_8 kernel and, since trnfft#100, the BF16 GEMM and the complex
# linear kernel load plainly and transpose on the Tensor engine: same bits.
# The default is on for the native path (Trn2/Trn3, where this was measured;
# Tensor-engine transposes are bit-exact from NeuronCore-v3 on) and in the
# simulator, off on the XLA path (Trn1: NeuronCore-v2 transposes are not
# bit-exact for NaN/Inf, and the rebuild was not measured there).
# ``TRNFFT_PE_TRANSPOSE=0``/``1`` (read per call) picks the kernel of each pair.
def _pe_transpose_enabled() -> bool:
    value = os.environ.get("TRNFFT_PE_TRANSPOSE", "").strip().lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    if _use_simulator() or _get_dispatch_path() != "xla":
        return True
    # XLA path: the load_transpose2d kernels, when nki defines them.
    return "_complex_gemm_t2d_kernel" not in globals()


def _gemm_kernel_names() -> tuple[str, str]:
    """(kernel attribute, op name) of the FP32 complex GEMM to launch."""
    if _pe_transpose_enabled():
        return "_complex_gemm_kernel", "complex_gemm"
    return "_complex_gemm_t2d_kernel", "complex_gemm_t2d"


def _bf16_kernel_names() -> tuple[str, str]:
    """(kernel attribute, op name) of the BF16 complex GEMM (trnfft#100)."""
    if _pe_transpose_enabled():
        return "_complex_gemm_kernel_bf16", "complex_gemm_bf16"
    return "_complex_gemm_bf16_t2d_kernel", "complex_gemm_bf16_t2d"


def _linear_kernel_names() -> tuple[str, str]:
    """(kernel attribute, op name) of the complex linear kernel (trnfft#100)."""
    if _pe_transpose_enabled():
        return "_complex_linear_kernel", "complex_linear"
    return "_complex_linear_t2d_kernel", "complex_linear_t2d"


# --- NKI kernels ---
# Ported from neuron-complex-ops/kernels.py (Apache 2.0)

if HAS_NKI:

    def _complex_gemm_tiles(a_real, a_imag, b_real, b_imag, c_real, c_imag, t2d, b_nk):
        """Tile loop shared by the complex GEMM / linear kernels.

        Writes ``C = A @ B`` (``b_nk=False``, B is (K, N)) or ``C = A @ B^T``
        (``b_nk=True``, B is (N, K) as ``nn.Linear`` stores it) into the HBM
        planes ``c_real`` / ``c_imag``:

            C_real = A_real @ B_real - A_imag @ B_imag
            C_imag = A_real @ B_imag + A_imag @ B_real

        ``nisa.nc_matmul`` contracts over the partition dim, so the stationary
        A tile must be K-major, (TILE_K, TILE_M); A is (M, K) row-major. With
        ``t2d=False`` each tile is read with a plain ``nl.load`` and transposed
        on the Tensor engine (``nisa.nc_transpose`` into PSUM, ``tensor_copy``
        to SBUF), and so is B when ``b_nk``. With ``t2d=True`` the tiles are
        read with ``nl.load_transpose2d``, as before trnfft#97. On Trn2 (SDK
        2.33) the runtime rebuilds a ``load_transpose2d`` input's IO DMA rings
        whenever its address changes, so every call with a fresh tensor paid
        for it. Tensor-engine transposes are bit-exact from NeuronCore-v3 on;
        the tiles, PSUM accumulation order and casts are the same in both
        modes, so they give the same bits.

        Tails (trnfft#100): M, K and N need not be multiples of their tiles.
        Full tiles run in ``affine_range`` loops; the last partial tile of
        each dimension is traced once more at its own static size. Until
        trnfft#100 the loops ran ``M // TILE_M`` (and ``K // TILE_K``,
        ``N // TILE_N``) times with no tail: at B = 1000, N = 64, rows
        896-999 were never written, and a K or N tail lost terms or columns.
        Shapes that are multiples of the tiles trace exactly as before.
        """
        M, K = a_real.shape
        N = b_real.shape[0] if b_nk else b_real.shape[1]

        TILE_M = min(M, PMAX)
        TILE_K = min(K, PMAX)
        TILE_N = min(N, 512)
        m_full, m_tail = M // TILE_M, M % TILE_M
        k_full, k_tail = K // TILE_K, K % TILE_K
        n_full, n_tail = N // TILE_N, N % TILE_N
        out_dtype = c_real.dtype
        # PSUM dtype of the Tensor-engine transposes: the input dtype from
        # NeuronCore-v3 on, FP32 only on v2 (a BF16 tile goes through FP32 and
        # back exactly there: the transpose multiplies by an identity).
        t_dtype = a_real.dtype
        if nisa.get_nc_version() == nisa.nc_version.gen2:
            t_dtype = nl.float32

        def _load_t(src, r_off, rows, c_off, cols):
            # src[r_off:+rows, c_off:+cols] as a (cols, rows) SBUF tile.
            # rows <= 128: one Tensor-engine transpose.
            if t2d:
                return nl.load_transpose2d(src[r_off : r_off + rows, c_off : c_off + cols])
            t = nl.load(src[r_off : r_off + rows, c_off : c_off + cols])
            pt = nl.ndarray((cols, rows), dtype=t_dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=pt, data=t, engine=nisa.engine.tensor)
            st = nl.ndarray((cols, rows), dtype=src.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=st, src=pt)
            return st

        def _load_b(src, k_off, tk, n_off, tn):
            # The (tk, tn) moving tile, partition dim = K.
            if not b_nk:
                return nl.load(src[k_off : k_off + tk, n_off : n_off + tn])
            if t2d:
                return nl.load_transpose2d(src[n_off : n_off + tn, k_off : k_off + tk])
            # B^T: up to 512 columns, transposed in blocks of 128 rows of B.
            bt = nl.ndarray((tk, tn), dtype=src.dtype, buffer=nl.sbuf)
            for c in range((tn + PMAX - 1) // PMAX):
                c0 = c * PMAX
                cs = min(PMAX, tn - c0)
                t = nl.load(src[n_off + c0 : n_off + c0 + cs, k_off : k_off + tk])
                pt = nl.ndarray((tk, cs), dtype=t_dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=pt, data=t, engine=nisa.engine.tensor)
                nisa.tensor_copy(dst=bt[:, c0 : c0 + cs], src=pt)
            return bt

        # B^T once per launch (trnfft#101 follow-up). Without this the
        # B^T path re-transposes (and re-negates) every W tile for every
        # 128-row tile of A: M/128 times. When the three K-major planes
        # (B_real^T, B_imag^T, -B_imag^T) fit in ``_HOIST_SBUF_BYTES`` per
        # partition they are built once, before the tile loops, with the same
        # loads, transposes and negation as ``_load_b``; the matmuls read
        # slices of them. Same values in the same matmuls: same bits.
        k_tiles = k_full + (1 if k_tail else 0)
        b_bytes = 4 if b_real.dtype == nl.float32 else 2
        hoist = b_nk and not t2d and 3 * k_tiles * N * b_bytes <= _HOIST_SBUF_BYTES
        if hoist:
            # Column block kt*N .. kt*N+N holds K-tile kt of B^T, (tk, N).
            bt_r = nl.ndarray((TILE_K, k_tiles * N), dtype=b_real.dtype, buffer=nl.sbuf)
            bt_i = nl.ndarray((TILE_K, k_tiles * N), dtype=b_real.dtype, buffer=nl.sbuf)
            bt_ni = nl.ndarray((TILE_K, k_tiles * N), dtype=b_real.dtype, buffer=nl.sbuf)
            for kt in range(k_tiles):
                k_off = kt * TILE_K
                tk = TILE_K if kt < k_full else k_tail
                for src, dst in ((b_real, bt_r), (b_imag, bt_i)):
                    for c in range((N + PMAX - 1) // PMAX):
                        c0 = c * PMAX
                        cs = min(PMAX, N - c0)
                        t = nl.load(src[c0 : c0 + cs, k_off : k_off + tk])
                        pt = nl.ndarray((tk, cs), dtype=t_dtype, buffer=nl.psum)
                        nisa.nc_transpose(dst=pt, data=t, engine=nisa.engine.tensor)
                        nisa.tensor_copy(dst=dst[:tk, kt * N + c0 : kt * N + c0 + cs], src=pt)
                # As nl.negative: multiply by -1 in the input dtype.
                nisa.tensor_scalar(
                    dst=bt_ni[:tk, kt * N : kt * N + N],
                    data=bt_i[:tk, kt * N : kt * N + N],
                    op0=nl.multiply,
                    operand0=-1.0,
                )

        def _tile(m_off, tm, n_off, tn):
            psum_cr = nl.zeros((tm, tn), dtype=nl.float32, buffer=nl.psum)
            psum_ci = nl.zeros((tm, tn), dtype=nl.float32, buffer=nl.psum)

            def _acc(k_off, tk, kt):
                ar_t = _load_t(a_real, m_off, tm, k_off, tk)
                ai_t = _load_t(a_imag, m_off, tm, k_off, tk)
                if hoist:
                    c0 = kt * N + n_off
                    br = bt_r[:tk, c0 : c0 + tn]
                    bi = bt_i[:tk, c0 : c0 + tn]
                    neg_bi = bt_ni[:tk, c0 : c0 + tn]
                else:
                    br = _load_b(b_real, k_off, tk, n_off, tn)
                    bi = _load_b(b_imag, k_off, tk, n_off, tn)
                    # NKI 2.24 doesn't support `psum -=` inside affine_range;
                    # negate B_imag on the Vector engine, then accumulate with +=.
                    neg_bi = nl.negative(bi)
                # C_real += A_real @ B_real  +  A_imag @ (-B_imag)
                nisa.nc_matmul(dst=psum_cr, stationary=ar_t, moving=br, accumulate=True)
                nisa.nc_matmul(dst=psum_cr, stationary=ai_t, moving=neg_bi, accumulate=True)
                # C_imag += A_real @ B_imag  +  A_imag @ B_real
                nisa.nc_matmul(dst=psum_ci, stationary=ar_t, moving=bi, accumulate=True)
                nisa.nc_matmul(dst=psum_ci, stationary=ai_t, moving=br, accumulate=True)

            for k in nl.affine_range(k_full):
                _acc(k * TILE_K, TILE_K, k)
            if k_tail:
                _acc(k_full * TILE_K, k_tail, k_full)

            # NKI 0.3.0: nl.copy returns a view; PSUM->SBUF goes through
            # nisa.tensor_copy into a pre-allocated SBUF tile. Cast after, when
            # the output is not FP32 (the BF16 kernel keeps the FP32 PSUM).
            cr_sbuf = nl.ndarray((tm, tn), dtype=nl.float32, buffer=nl.sbuf)
            ci_sbuf = nl.ndarray((tm, tn), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cr_sbuf, src=psum_cr)
            nisa.tensor_copy(dst=ci_sbuf, src=psum_ci)
            if out_dtype != nl.float32:
                cr_sbuf = nl.cast(cr_sbuf, dtype=out_dtype)
                ci_sbuf = nl.cast(ci_sbuf, dtype=out_dtype)
            nl.store(c_real[m_off : m_off + tm, n_off : n_off + tn], value=cr_sbuf)
            nl.store(c_imag[m_off : m_off + tm, n_off : n_off + tn], value=ci_sbuf)

        for m in nl.affine_range(m_full):
            for n in nl.affine_range(n_full):
                _tile(m * TILE_M, TILE_M, n * TILE_N, TILE_N)
            if n_tail:
                _tile(m * TILE_M, TILE_M, n_full * TILE_N, n_tail)
        if m_tail:
            for n in nl.affine_range(n_full):
                _tile(m_full * TILE_M, m_tail, n * TILE_N, TILE_N)
            if n_tail:
                _tile(m_full * TILE_M, m_tail, n_full * TILE_N, n_tail)

    def _gemm_out(a_real, b_real, dtype, b_nk=False):
        n = b_real.shape[0] if b_nk else b_real.shape[1]
        shape = (a_real.shape[0], n)
        return (
            nl.ndarray(shape, dtype=dtype, buffer=nl.shared_hbm),
            nl.ndarray(shape, dtype=dtype, buffer=nl.shared_hbm),
        )

    @nki.jit
    def _complex_gemm_kernel(a_real, a_imag, b_real, b_imag):
        """Complex GEMM: C = A @ B, with A (M, K), B (K, N) as real/imag planes.

        4 real matmuls into 2 FP32 PSUM tiles per (128, 512) output tile; the
        result is cast to the input dtype. A is read with plain loads and
        transposed on the Tensor engine (trnfft#97), see
        ``_complex_gemm_tiles``. Any M, K, N (tail tiles since trnfft#100).
        ``_complex_gemm_t2d_kernel`` is the ``load_transpose2d`` version
        (``TRNFFT_PE_TRANSPOSE=0``, and the XLA path); same bits.
        """
        c_real, c_imag = _gemm_out(a_real, b_real, a_real.dtype)
        _complex_gemm_tiles(a_real, a_imag, b_real, b_imag, c_real, c_imag, False, False)
        return c_real, c_imag

    @nki.jit
    def _complex_gemm_t2d_kernel(a_real, a_imag, b_real, b_imag):
        """``_complex_gemm_kernel`` reading A with ``nl.load_transpose2d``.

        Kept behind ``TRNFFT_PE_TRANSPOSE=0`` and as the XLA-path (Trn1)
        default. Same tiles and PSUM order, so the same bits.
        """
        c_real, c_imag = _gemm_out(a_real, b_real, a_real.dtype)
        _complex_gemm_tiles(a_real, a_imag, b_real, b_imag, c_real, c_imag, True, False)
        return c_real, c_imag

    @nki.jit
    def _complex_gemm_tn_kernel(at_real, at_imag, b_real, b_imag):
        """Stacked complex GEMMs with the left operand given transposed (trnfft#97).

        ``at`` is (G*K, M): rows ``g*K .. g*K+K`` hold ``A_g^T`` (K, M).
        ``b`` is (K, N). Returns ``c`` (G*M, N) with rows ``g*M .. g*M+M``
        equal to ``A_g @ B``, as two FP32 planes.

        Same tiles, PSUM accumulation order and casts as
        ``_complex_gemm_kernel`` on the (G*M, K) matrix of the stacked
        ``A_g``, so the result is bit-identical to it. The difference is the
        load: the stationary tile is already K-major in HBM, so it is a plain
        ``nl.load`` of contiguous rows instead of ``nl.load_transpose2d``.
        On Trn2 (SDK 2.33) the runtime rebuilds a transposed load's IO DMA
        rings whenever the input address changes (``kbl_exec_pre``: ~4.7 MB of
        descriptors copied in, 2-9 ms for an 8 MiB plane); a plain load's
        rings are small. The six-step's first pass at N = 2^13 reads the
        (B, N2, N1) view of x directly in this layout, which also drops the
        transpose before it.
        """
        GK, M = at_real.shape
        K, N = b_real.shape
        G = GK // K
        assert G * K == GK, f"rows {GK} not a multiple of K={K}"

        TILE_M = min(M, 128)
        TILE_K = min(K, 128)
        TILE_N = min(N, 512)
        # No tail tiles here: the six-step guard (_six_step_pass1_tn_ok) only
        # launches whole tiles. Refuse other shapes rather than drop the tail
        # (trnfft#100).
        assert M % TILE_M == 0 and K % TILE_K == 0 and N % TILE_N == 0, (
            f"_complex_gemm_tn_kernel needs M, K <= 128 or multiples of 128 and "
            f"N <= 512 or a multiple of 512; got M={M}, K={K}, N={N}"
        )

        c_real = nl.ndarray((G * M, N), dtype=at_real.dtype, buffer=nl.shared_hbm)
        c_imag = nl.ndarray((G * M, N), dtype=at_real.dtype, buffer=nl.shared_hbm)

        for g in nl.affine_range(G):
            for m in nl.affine_range(M // TILE_M):
                for n in nl.affine_range(N // TILE_N):
                    m_off = m * TILE_M
                    n_off = n * TILE_N
                    c_off = g * M + m_off

                    psum_cr = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                    psum_ci = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                    for k in nl.affine_range(K // TILE_K):
                        r_off = g * K + k * TILE_K
                        k_off = k * TILE_K
                        # (TILE_K, TILE_M), partition dim = K: no transpose.
                        ar_t = nl.load(at_real[r_off : r_off + TILE_K, m_off : m_off + TILE_M])
                        ai_t = nl.load(at_imag[r_off : r_off + TILE_K, m_off : m_off + TILE_M])
                        br = nl.load(b_real[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                        bi = nl.load(b_imag[k_off : k_off + TILE_K, n_off : n_off + TILE_N])
                        neg_bi = nl.negative(bi)
                        nisa.nc_matmul(dst=psum_cr, stationary=ar_t, moving=br, accumulate=True)
                        nisa.nc_matmul(dst=psum_cr, stationary=ai_t, moving=neg_bi, accumulate=True)
                        nisa.nc_matmul(dst=psum_ci, stationary=ar_t, moving=bi, accumulate=True)
                        nisa.nc_matmul(dst=psum_ci, stationary=ai_t, moving=br, accumulate=True)

                    cr_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                    ci_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=cr_sbuf, src=psum_cr)
                    nisa.tensor_copy(dst=ci_sbuf, src=psum_ci)
                    if at_real.dtype != nl.float32:
                        cr_sbuf = nl.cast(cr_sbuf, dtype=at_real.dtype)
                        ci_sbuf = nl.cast(ci_sbuf, dtype=at_real.dtype)
                    nl.store(c_real[c_off : c_off + TILE_M, n_off : n_off + TILE_N], value=cr_sbuf)
                    nl.store(c_imag[c_off : c_off + TILE_M, n_off : n_off + TILE_N], value=ci_sbuf)

        return c_real, c_imag

    @nki.jit
    def _complex_gemm_kernel_bf16(a_real, a_imag, b_real, b_imag):
        """BF16 Complex GEMM: BF16 inputs → FP32 PSUM → FP32 output.

        Architectural pattern: nc_matmul accumulates into FP32 PSUM regardless
        of input dtype. By allocating output as FP32 and *not* casting the PSUM
        back to BF16, we keep the full FP32 accumulation quality while
        computing in BF16 (≈2× Tensor Engine throughput).

        This is the "PSUM is a free FP32 accumulator" principle:
          BF16 compute speed × FP32 accumulation precision.

        Inputs a_real, a_imag, b_real, b_imag are expected to be BF16.
        Output c_real, c_imag are always FP32 — the PSUM result, not rounded.

        Same tiles as ``_complex_gemm_kernel`` (any M, K, N; plain loads and
        Tensor-engine transposes of A, trnfft#97/#100); the only difference
        is the FP32 output with no cast. ``_complex_gemm_bf16_t2d_kernel`` is
        the ``load_transpose2d`` version (``TRNFFT_PE_TRANSPOSE=0``, XLA path).
        """
        c_real, c_imag = _gemm_out(a_real, b_real, nl.float32)
        _complex_gemm_tiles(a_real, a_imag, b_real, b_imag, c_real, c_imag, False, False)
        return c_real, c_imag

    @nki.jit
    def _complex_gemm_bf16_t2d_kernel(a_real, a_imag, b_real, b_imag):
        """``_complex_gemm_kernel_bf16`` reading A with ``nl.load_transpose2d``
        (the pre-trnfft#100 load; ``TRNFFT_PE_TRANSPOSE=0`` and the XLA path)."""
        c_real, c_imag = _gemm_out(a_real, b_real, nl.float32)
        _complex_gemm_tiles(a_real, a_imag, b_real, b_imag, c_real, c_imag, True, False)
        return c_real, c_imag

    @nki.jit
    def _complex_linear_kernel(x_real, x_imag, w_real, w_imag):
        """Complex linear layer: y = x @ W^T where x and W are complex.

        Equivalent to:
            y_re = x_re @ W_re.T - x_im @ W_im.T
            y_im = x_re @ W_im.T + x_im @ W_re.T

        Shapes:
            x_real, x_imag: (M, K_in)         — activations
            w_real, w_imag: (K_out, K_in)     — weights (stored row-major)
            returns y_real, y_imag: (M, K_out)

        A complex GEMM with W^T as the moving operand: x tiles are the
        stationary (transposed so partition dim = K_in), W tiles the moving
        (transposed so partition dim = K_in). Both are read with plain loads
        and transposed on the Tensor engine (trnfft#100, as #97 did for the
        DFT-GEMM); any M, K_in, K_out. ``_complex_linear_t2d_kernel`` is the
        ``load_transpose2d`` version (``TRNFFT_PE_TRANSPOSE=0``, XLA path).
        """
        y_real, y_imag = _gemm_out(x_real, w_real, x_real.dtype, b_nk=True)
        _complex_gemm_tiles(x_real, x_imag, w_real, w_imag, y_real, y_imag, False, True)
        return y_real, y_imag

    @nki.jit
    def _complex_linear_t2d_kernel(x_real, x_imag, w_real, w_imag):
        """``_complex_linear_kernel`` reading x and W with ``nl.load_transpose2d``
        (the pre-trnfft#100 load; ``TRNFFT_PE_TRANSPOSE=0`` and the XLA path)."""
        y_real, y_imag = _gemm_out(x_real, w_real, x_real.dtype, b_nk=True)
        _complex_gemm_tiles(x_real, x_imag, w_real, w_imag, y_real, y_imag, True, True)
        return y_real, y_imag

    @nki.jit
    def _complex_mul_kernel(a_real, a_imag, b_real, b_imag):
        """Fused element-wise complex multiply.

        Computes (a_re + i a_im) * (b_re + i b_im)
          = (a_re*b_re - a_im*b_im) + i (a_re*b_im + a_im*b_re)
        in a single kernel, avoiding 6 separate HBM round-trips.

        NKI 2.24 partition-dim constraint: any SBUF tile must be 2D with first
        dim being the partition dim, and partition size ≤ 128. We reshape the
        flat input into (PMAX, total/PMAX) and process column tiles of at most
        512. Inputs must have total size divisible by PMAX (128); the
        dispatcher falls back to PyTorch otherwise.
        """
        shape = a_real.shape
        total = 1
        for s in shape:
            total *= s

        # Trainium's Vector Engine partition limit. Inputs that are not multiples
        # of 128 elements would need a partition tail; the dispatcher falls back.
        assert total % 128 == 0, (
            f"_complex_mul_kernel requires total size divisible by 128; got {total}"
        )

        free = total // 128
        # Free-dim tiles of FMAX columns in an affine_range, plus one tail tile
        # of free % FMAX columns traced at its own static size (trnfft#100;
        # before, free > FMAX had to be a multiple of FMAX).
        FMAX = 512
        free_tile = min(free, FMAX)
        n_tiles, f_tail = free // free_tile, free % free_tile

        c_real = nl.ndarray(shape, dtype=a_real.dtype, buffer=nl.shared_hbm)
        c_imag = nl.ndarray(shape, dtype=a_real.dtype, buffer=nl.shared_hbm)

        # Reshape views: flatten then expose as (128, free).
        a_re_2d = a_real.reshape((128, free))
        a_im_2d = a_imag.reshape((128, free))
        b_re_2d = b_real.reshape((128, free))
        b_im_2d = b_imag.reshape((128, free))
        c_re_2d = c_real.reshape((128, free))
        c_im_2d = c_imag.reshape((128, free))

        def _cols(f_off, width):
            f_end = f_off + width  # constant slice size; no min()
            ar = nl.load(a_re_2d[:, f_off:f_end])
            ai = nl.load(a_im_2d[:, f_off:f_end])
            br = nl.load(b_re_2d[:, f_off:f_end])
            bi = nl.load(b_im_2d[:, f_off:f_end])

            # NKI 0.3.0: Python `*`/`+`/`-` operators aren't defined on
            # NkiTensor; use nl.multiply / nl.subtract / nl.add explicitly.
            cr = nl.subtract(nl.multiply(ar, br), nl.multiply(ai, bi))
            ci = nl.add(nl.multiply(ar, bi), nl.multiply(ai, br))

            nl.store(c_re_2d[:, f_off:f_end], value=cr)
            nl.store(c_im_2d[:, f_off:f_end], value=ci)

        for t in nl.affine_range(n_tiles):
            _cols(t * free_tile, free_tile)
        if f_tail:
            _cols(n_tiles * free_tile, f_tail)

        return c_real, c_imag


# --- NKI kernel wrappers ---


def _to_xla(*tensors):
    """Move a list of tensors to the XLA device."""
    import torch_xla

    device = torch_xla.device()
    return [t.to(device) for t in tensors], tensors[0].device


# ── Device dispatch path: native TorchNeuron vs XLA ──────────────────────────
#
# Two Neuron PyTorch lineages coexist:
#   native  torch-neuronx 2.14+ (TorchNeuron, Trn2/Trn3): Trainium is a
#           PyTorch `neuron` device; kernels are custom ops registered via
#           `torch_neuronx.nki_op` (see `_native.py`).
#   xla     torch-neuronx 2.9 lineage (PyTorch/XLA; Trn1 and Inf2 stay
#           here): kernels dispatch on the torch_xla device via `_to_xla()`.
# Anything else is "cpu": no device path, so a kernel launch outside the
# simulator goes through `_to_xla` and raises exactly as it did before the
# native path existed. Detection runs on the first launch, not at import,
# and the native probe never imports torch_xla.

_NEURON_DEVICE = "neuron"
_dispatch_path: str | None = None
_native_ops: dict | None = None
_xla_deprecation_warned = False

_XLA_DEPRECATION_MESSAGE = (
    "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
    "Trn1/Inf2 remain supported in maintenance mode."
)


def _native_available() -> bool:
    try:
        import torch_neuronx
    except Exception:
        return False
    # The XLA-lineage torch_neuronx has no `nki_op`.
    if not hasattr(torch_neuronx, "nki_op"):
        return False
    try:
        torch.device(_NEURON_DEVICE)  # torch_neuronx renames PrivateUse1 to "neuron"
        return torch_neuronx.device_count() > 0
    except Exception:
        return False


def _torch_xla_installed() -> bool:
    try:
        return importlib.util.find_spec("torch_xla") is not None
    except (ImportError, ValueError):
        # find_spec raises ValueError for a module in sys.modules whose
        # __spec__ is None (a stand-in module); it is importable.
        return "torch_xla" in sys.modules


def _detect_dispatch_path() -> str:
    # TRNSCI_NKI_DISPATCH=auto|native|xla|cpu is shared by every trnsci
    # library, so one variable pins the whole suite on hosts with both lineages.
    forced = os.environ.get("TRNSCI_NKI_DISPATCH", "auto").strip().lower()
    if forced == "native":
        if not _native_available():
            raise RuntimeError(
                "TRNSCI_NKI_DISPATCH=native, but no TorchNeuron backend with a "
                "'neuron' device is available"
            )
        return "native"
    if forced == "xla":
        if not _torch_xla_installed():
            raise RuntimeError("TRNSCI_NKI_DISPATCH=xla, but torch_xla is not installed")
        return "xla"
    if forced == "cpu":
        return "cpu"
    if forced != "auto":
        raise ValueError(f"TRNSCI_NKI_DISPATCH must be auto, native, xla, or cpu; got {forced!r}")
    if _native_available():
        return "native"
    if _torch_xla_installed():
        return "xla"
    return "cpu"


def _maybe_warn_xla_deprecation() -> None:
    # Dormant: the suite turns this on at TorchNeuron GA (trnsci#35).
    global _xla_deprecation_warned
    if _xla_deprecation_warned or os.environ.get("TRNSCI_XLA_DEPRECATION", "") != "1":
        return
    _xla_deprecation_warned = True
    warnings.warn(_XLA_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def _get_dispatch_path() -> str:
    """Return "native", "xla", or "cpu" (cached after the first call)."""
    global _dispatch_path
    if _dispatch_path is None:
        _dispatch_path = _detect_dispatch_path()
        if _dispatch_path == "xla":
            _maybe_warn_xla_deprecation()
    return _dispatch_path


def _is_native() -> bool:
    """True when kernels launch as native TorchNeuron custom ops."""
    return _get_dispatch_path() == "native"


def _get_native_ops() -> dict:
    """Import `_native` (registering the trnfft:: custom ops) on first use."""
    global _native_ops
    if _native_ops is None:
        from . import _native

        _native_ops = _native.OPS
    return _native_ops


def _kernel_device() -> torch.device:
    """Device the multi-stage drivers keep their operands on between launches."""
    if _is_native():
        return torch.device(_NEURON_DEVICE)
    import torch_xla

    return torch_xla.device()


# ── Host results through the caching host allocator ─────────────────────────
#
# Same fix as trnblas#61/#64 and trnrand#70. On Trn2 (SDK 2.33, native
# TorchNeuron), a device->host copy into fresh pageable memory (`.cpu()`,
# `torch.empty` + `copy_`) runs at ~4 GB/s, because the host page-faults every
# page it touches. A copy into memory that is already faulted in runs at the
# link rate, 13-14 GB/s. `torch.empty(..., pin_memory=True)` comes from the
# accelerator's caching host allocator: each call still returns a fresh tensor
# (a block is recycled only after the tensor that owned it is freed, so results
# never alias), and recycled blocks come back already faulted in. An FFT
# result is a real and an imaginary plane, so each result is two such copies.
#
# Costs: blocks are rounded up to a power of two, and freed blocks stay cached
# until `torch.accelerator.empty_host_cache()`. TRNFFT_PINNED_OUTPUT=0 (read
# per call) turns this off (plain pageable memory). Planes under
# `_PINNED_OUTPUT_MIN_BYTES` always use pageable memory. The XLA path, the
# simulator and torch.compile tracing are unchanged. The values are the same
# bits either way.
_PINNED_OUTPUT_MIN_BYTES = 1 << 20


def _pinned_output_enabled() -> bool:
    value = os.environ.get("TRNFFT_PINNED_OUTPUT", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _neuron_host_allocator() -> bool:
    """True when `pin_memory=True` would come from the Neuron accelerator's
    caching host allocator. On a CUDA build with no GPU, pinning raises, so
    the cache is used only when the current accelerator is Neuron."""
    acc = getattr(torch, "accelerator", None)
    if acc is None or not acc.is_available():
        return False
    return acc.current_accelerator().type == "neuron"


def _use_host_cache(device, nbytes: int) -> bool:
    """True when a CPU result of `nbytes` should come from the caching host
    allocator: native eager path, CPU destination, large enough, not disabled."""
    return (
        torch.device(device).type == "cpu"
        and nbytes >= _PINNED_OUTPUT_MIN_BYTES
        and _pinned_output_enabled()
        and not _use_simulator()
        and _is_native()
        and not torch.compiler.is_compiling()
        and _neuron_host_allocator()
    )


def _host_empty(shape, dtype: torch.dtype, device="cpu") -> torch.Tensor:
    """`torch.empty(shape, dtype=dtype, device=device)`; a large CPU tensor on the
    native path comes from the caching host allocator (fresh, already faulted in)."""
    shape = tuple(shape)
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if _use_host_cache(device, nbytes):
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.empty(shape, dtype=dtype, device=device)


def _to_host(t: torch.Tensor, device="cpu") -> torch.Tensor:
    """`t.to(device)`. A large device -> CPU copy lands in a fresh tensor from
    the caching host allocator, about 3x faster than `.cpu()` (trnblas#61)."""
    if t.device.type == "cpu" or not _use_host_cache(device, t.numel() * t.element_size()):
        return t.to(device)
    return _host_empty(t.shape, t.dtype, device).copy_(t)


_LAUNCH_LOCK = threading.Lock()


def _launch(kernel, op_name: str, *args):
    """Run one kernel launch on the Neuron device; results return to the caller's device.

    ``args`` are the kernel's positional arguments: tensors and, for the
    butterfly kernels, Python ints. On the native path the tensors move to
    ``device="neuron"`` and the ``trnfft::<op_name>`` custom op runs; the XLA
    path moves them with ``_to_xla`` and calls ``kernel`` directly. Tensors
    already on the kernel device (the Stockham / butterfly drivers keep them
    there between stages) are not copied, and the result stays there.
    """
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    if _is_native():
        orig = tensors[0].device
        op = _get_native_ops()[op_name]
        # A tensor already on ``neuron:k`` stays there (``.to("neuron")`` is a
        # no-op), so per-core shards launch on their own core (trnfft#54).
        moved = [a.to(_NEURON_DEVICE) if isinstance(a, torch.Tensor) else a for a in args]
        # NKI tracing on a dispatch-cache miss is not thread-safe (trnsolver#54);
        # the call only enqueues work, so serialising it keeps cores concurrent.
        with _LAUNCH_LOCK:
            out = op(*moved)
    else:
        moved, orig = _to_xla(*tensors)
        it = iter(moved)
        out = kernel(*(next(it) if isinstance(a, torch.Tensor) else a for a in args))
    if isinstance(out, tuple | list):
        return tuple(_to_host(o, orig) for o in out)
    return _to_host(out, orig)


def _simulate_kernel(kernel, *tensors):
    """Route a kernel call through `nki.simulate` on CPU.

    Converts each input tensor to numpy, calls the simulator, and marshals
    results back to torch tensors on the original device. Kernels with
    multiple HBM outputs return a tuple of numpy arrays; single-output
    kernels return a numpy array.

    Used by the dispatch wrappers when ``TRNFFT_USE_SIMULATOR=1`` is set
    in the environment. Bypasses torch_xla and NEFF compile entirely —
    correctness iteration only; hardware still owns perf numbers.
    """
    orig_device = tensors[0].device
    # numpy has no BF16 dtype — cast to FP32 before simulation (simulator
    # uses the kernel's declared dtype, not the numpy input dtype).
    np_args = [
        t.detach().cpu().float().numpy() if t.dtype == torch.bfloat16 else t.detach().cpu().numpy()
        for t in tensors
    ]
    out = nki.simulate(kernel)(*np_args)
    if isinstance(out, tuple):
        return tuple(torch.from_numpy(np.asarray(o)).to(orig_device) for o in out)
    return torch.from_numpy(np.asarray(out)).to(orig_device)


def _nki_complex_gemm(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """NKI complex GEMM with stationary reuse (autograd-aware)."""
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    from .autograd import complex_gemm_autograd

    c_real, c_imag = complex_gemm_autograd(a.real, a.imag, b.real, b.imag)
    return ComplexTensor(c_real, c_imag)


def complex_gemm_bf16(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """BF16 complex GEMM: BF16 inputs → FP32 PSUM → FP32 output.

    On NKI hardware: uses ``_complex_gemm_kernel_bf16`` which skips the
    PSUM→BF16 cast, preserving FP32 accumulation quality at BF16 throughput.
    On CPU (no NKI): falls back to ``complex_matmul`` with whatever dtype
    the inputs carry (correctness path; no throughput benefit on CPU).
    """
    if _use_nki() and not _use_simulator():
        # Move inputs to the device, then move results back to the caller's
        # device (like the autograd wrappers). Returning device tensors here
        # made _fft_via_ozaki_hq mix device and CPU tensors (trnfft#80).
        kernel_name, op_name = _bf16_kernel_names()
        c_real, c_imag = _launch(
            globals().get(kernel_name), op_name, a.real, a.imag, b.real, b.imag
        )
        return ComplexTensor(c_real, c_imag)
    if _use_simulator():
        kernel = globals()[_bf16_kernel_names()[0]]
        c_real, c_imag = _simulate_kernel(kernel, a.real, a.imag, b.real, b.imag)
        return ComplexTensor(c_real, c_imag)
    # CPU fallback: simulate FP32 PSUM behaviour by casting BF16 inputs to
    # FP32 before the matmul. On hardware, nc_matmul accumulates BF16 products
    # into a FP32 PSUM; on CPU torch.matmul accumulates in BF16 without the
    # FP32 PSUM, giving ~50% error at N=64. Casting to FP32 first matches the
    # hardware's FP32 accumulation quality at the cost of BF16 quantisation of
    # W and x (which is the intended accuracy regime: ~2e-3 rel error, u_bf16).
    a_fp32 = ComplexTensor(a.real.float(), a.imag.float())
    b_fp32 = ComplexTensor(b.real.float(), b.imag.float())
    result = complex_matmul(a_fp32, b_fp32)
    return ComplexTensor(result.real, result.imag)  # already FP32


# _ozaki_split_bf16 and complex_gemm_ozaki were defined here in v0.18 but have been
# consolidated into trnfft/fft_core.py in v0.19. The split helpers (_ozaki_split_bf16,
# _ozaki_split_3way_bf16) live in fft_core.py; complex_gemm_ozaki was superseded by
# the direct _fft_via_ozaki / _fft_via_ozaki_hq drivers in fft_core.py.


def _nki_complex_linear(
    x: ComplexTensor, w_real: torch.Tensor, w_imag: torch.Tensor
) -> ComplexTensor:
    """NKI complex linear: y = x @ W^T (complex) via fused 4-matmul kernel (autograd-aware)."""
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    from .autograd import complex_linear_autograd

    # The kernel is fp32. float64 host tensors reached it as is and came back
    # inf/NaN (#131): run it in fp32 and return float64, as complex_gemm does.
    out_dtype = torch.promote_types(x.real.dtype, w_real.dtype)
    if out_dtype == torch.float64:
        xr, xi, wr, wi = (t.float() for t in (x.real, x.imag, w_real, w_imag))
        y_real, y_imag = complex_linear_autograd(xr, xi, wr, wi)
        return ComplexTensor(y_real.to(out_dtype), y_imag.to(out_dtype))
    y_real, y_imag = complex_linear_autograd(x.real, x.imag, w_real, w_imag)
    return ComplexTensor(y_real, y_imag)


def _nki_complex_mask(mask: ComplexTensor, spec: ComplexTensor) -> ComplexTensor:
    """NKI fused complex mask application (autograd-aware).

    The kernel requires total element count divisible by 128 (the Trainium
    Vector Engine partition limit). For inputs that aren't, fall back to
    PyTorch element-wise multiply (which is already autograd-safe).
    """
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    total = mask.real.numel()
    if total % 128 != 0:
        return mask * spec
    # Force contiguous layout before the kernel's reshape to (128, free).
    # Non-contiguous inputs (e.g., from broadcasting or transposed views) break
    # the reshape at NKI compile time even when the logical shape is fine.
    from .autograd import complex_mul_autograd

    c_real, c_imag = complex_mul_autograd(
        mask.real.contiguous(),
        mask.imag.contiguous(),
        spec.real.contiguous(),
        spec.imag.contiguous(),
    )
    return ComplexTensor(c_real, c_imag)
