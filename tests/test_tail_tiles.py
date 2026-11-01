"""Tail tiles on device (trnfft#100).

Every NKI kernel at batch / row counts that are not multiples of its 128-row
tile (B = 129, 1000, 4097), and the GEMM-family kernels at K and N that are
not multiples of their 128 / 512 tiles, against ``torch.fft`` / matmul in
FP64. Before trnfft#99/#100 the DFT-GEMM, the BF16 GEMM and the complex
linear kernel ran ``M // 128`` row tiles with no tail, so the last partial
tile was never written (B = 1000, N = 64: rows 896-999), and a K or N tail
lost terms or columns. Rows are checked one by one: a garbage row has
relative error ~1, far above every tolerance here.

Inputs are built on the host (device ``torch.randn`` is shape-sensitive,
aws-neuron-sdk#1439) and results come back to the host.
"""

from __future__ import annotations

import importlib

import pytest
import torch

import trnfft
from trnfft import ComplexTensor
from trnfft import fft_core as fc

pytestmark = pytest.mark.neuron

_precision = importlib.import_module("trnfft.precision")

BATCHES = [129, 1000, 4097]


def _rand_pair(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)


def _row_err(got: ComplexTensor, want: torch.Tensor) -> torch.Tensor:
    """Per-row max-norm relative error of ``got`` against the FP64 ``want``."""
    g = torch.complex(got.real.detach().cpu().double(), got.imag.detach().cpu().double())
    g = g.reshape(want.shape[0], -1)
    w = want.reshape(want.shape[0], -1)
    return (g - w).abs().amax(dim=1) / w.abs().amax(dim=1).clamp_min(1e-30)


def _check_rows(got, want, tol):
    err = _row_err(got, want)
    bad = (err > tol).nonzero().flatten()
    assert bad.numel() == 0, (
        f"{bad.numel()} of {err.numel()} rows over {tol:g}: first {bad[:4].tolist()}, "
        f"max {err.max().item():.3g}"
    )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        a.contiguous().cpu().view(torch.int32), b.contiguous().cpu().view(torch.int32)
    )


def _fft_ref(xr, xi, inverse=False):
    x = torch.complex(xr.double(), xi.double())
    return torch.fft.ifft(x) if inverse else torch.fft.fft(x)


# id -> (n, callable(ComplexTensor) -> ComplexTensor, tolerance, setup(monkeypatch)).
# Each one reaches a different kernel (or kernel mix).
def _paths():
    def env(**kv):
        def setup(mp):
            for k, v in kv.items():
                mp.setenv(k, v)

        return setup

    def butterfly(mp):
        # Threshold 0 turns off every GEMM path; "kahan" skips Stockham.
        mp.setattr(fc, "_DFT_GEMM_THRESHOLD", 0)

    none = env()
    fft = trnfft.fft
    return {
        # _complex_gemm_kernel, packed (g = 2 rows per GEMM row at N = 64).
        "dft-gemm-64": (64, fft, 1e-5, none),
        # _complex_gemm_kernel, K = N = 256 (two K tiles), unpacked.
        "dft-gemm-256": (256, fft, 1e-5, none),
        # _complex_gemm_kernel unpacked at N = 64 (TRNFFT_GEMM_PACK=0).
        "dft-gemm-64-nopack": (64, fft, 1e-5, env(TRNFFT_GEMM_PACK="0")),
        # _complex_gemm_t2d_kernel (TRNFFT_PE_TRANSPOSE=0).
        "dft-gemm-64-t2d": (64, fft, 1e-5, env(TRNFFT_PE_TRANSPOSE="0")),
        # four_step_fft_kernel (batch-fused four-step).
        "four-step-fused-4096": (4096, fft, 1e-5, none),
        # Unfused four-step: _complex_gemm_kernel x2 + _complex_mul_kernel.
        "four-step-2048": (2048, lambda x: fc._fft_via_four_step(x, False), 1e-5, none),
        # Radix-4 Stockham stage kernel.
        "stockham-r4-256": (256, lambda x: fc._fft_via_stockham_nki(x, False), 1e-5, none),
        # Radix-8 W_8 kernel (plain load + PE transpose).
        "stockham-r8-512": (512, lambda x: fc._fft_via_stockham_nki_r8(x, False), 1e-5, none),
        # Radix-8 W_8 kernel, load_transpose2d version.
        "stockham-r8-512-t2d": (
            512,
            lambda x: fc._fft_via_stockham_nki_r8(x, False),
            1e-5,
            env(TRNFFT_PE_TRANSPOSE="0"),
        ),
        # Mixed radix-8 / radix-4 Stockham.
        "stockham-mixed-2048": (
            2048,
            lambda x: fc._fft_via_stockham_nki_mixed(x, False),
            1e-5,
            none,
        ),
        # Kahan butterfly stage kernel.
        "butterfly-kahan-64": (
            64,
            lambda x: fc._cooley_tukey_nki_nograd(x, False, precision="kahan"),
            1e-5,
            butterfly,
        ),
        # BF16 precision modes: _complex_gemm_kernel_bf16 (+ the FP32 GEMM).
        "bf16-64": (64, lambda x: fc._fft_via_gemm_bf16(x, False), 2e-2, none),
        "bf16-64-t2d": (
            64,
            lambda x: fc._fft_via_gemm_bf16(x, False),
            2e-2,
            env(TRNFFT_PE_TRANSPOSE="0"),
        ),
        "bf16-256": (256, lambda x: fc._fft_via_gemm_bf16(x, False), 2e-2, none),
        "bf16-refined-64": (64, lambda x: fc._fft_iterative_refinement(x, False), 1e-4, none),
        "ozaki-64": (64, lambda x: fc._fft_via_ozaki(x, False), 1e-3, none),
        "ozaki-hq-64": (64, lambda x: fc._fft_via_ozaki_hq(x, False), 1e-3, none),
    }


_PATHS = _paths()
# The six-step at N = 2^13 is 64 MiB per plane at B = 1000; B = 4097 is left out.
_SIX_STEP_BATCHES = [129, 1000]
# The Stockham drivers run one launch per stage plus host permutes; at
# B = 4097 the radix-8 / mixed paths take over 10 minutes on Trn2, so they
# stop at B = 1000 (they pad any batch to a multiple of 128 anyway).
_CASES = [
    (path, batch)
    for path in sorted(_PATHS)
    for batch in BATCHES
    if not (path.startswith("stockham") and batch > 1000)
]


@pytest.mark.parametrize("path,batch", _CASES)
def test_fft_path_ragged_batch(nki_backend, monkeypatch, path, batch):
    n, fn, tol, setup = _PATHS[path]
    setup(monkeypatch)
    monkeypatch.setattr(_precision, "_precision", "fast")
    xr, xi = _rand_pair(batch, n, seed=batch * 31 + n)
    y = fn(ComplexTensor(xr, xi))
    assert tuple(y.real.shape) == (batch, n)
    _check_rows(y, _fft_ref(xr, xi), tol)


@pytest.mark.parametrize("batch", _SIX_STEP_BATCHES)
def test_six_step_ragged_batch(nki_backend, monkeypatch, batch):
    monkeypatch.setattr(_precision, "_precision", "fast")
    n = 1 << 13
    xr, xi = _rand_pair(batch, n, seed=batch)
    y = fc._fft_via_six_step(ComplexTensor(xr, xi), False)
    _check_rows(y, _fft_ref(xr, xi), 1e-5)


@pytest.mark.parametrize("batch", BATCHES)
def test_bluestein_ragged_batch(nki_backend, monkeypatch, batch):
    """N = 100 pads to 256: three DFT-GEMM sub-FFTs plus chirp multiplies."""
    monkeypatch.setattr(_precision, "_precision", "fast")
    xr, xi = _rand_pair(batch, 100, seed=batch + 100)
    y = trnfft.fft(ComplexTensor(xr, xi))
    _check_rows(y, _fft_ref(xr, xi), 1e-4)


# (M, K, N): tails in M (> 128, not a multiple), K (> 128, not a multiple)
# and N (> 512, not a multiple), and small odd sizes.
_GEMM_SHAPES = [(1000, 64, 64), (129, 128, 128), (200, 200, 600), (7, 300, 5), (4097, 8, 8)]


def _gemm_inputs(m, k, n, seed, b_nk=False, dtype=torch.float32):
    ar, ai = _rand_pair(m, k, seed=seed)
    br, bi = _rand_pair(*((n, k) if b_nk else (k, n)), seed=seed + 1)
    return [t.to(dtype) for t in (ar, ai, br, bi)]


def _gemm_ref(ar, ai, br, bi, b_nk=False):
    a = torch.complex(ar.double(), ai.double())
    b = torch.complex(br.double(), bi.double())
    return a @ (b.T if b_nk else b)


def _both_kernels(monkeypatch, f):
    """``f()`` with the load_transpose2d kernels (=0), then the plain-load ones (=1)."""
    monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "0")
    old = f()
    monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "1")
    new = f()
    return old, new


@pytest.mark.parametrize("m,k,n", _GEMM_SHAPES)
def test_complex_gemm_tails(nki_backend, monkeypatch, m, k, n):
    from trnfft.nki.dispatch import complex_gemm

    ar, ai, br, bi = _gemm_inputs(m, k, n, seed=m + k + n)
    old, new = _both_kernels(
        monkeypatch, lambda: complex_gemm(ComplexTensor(ar, ai), ComplexTensor(br, bi))
    )
    _check_rows(new, _gemm_ref(ar, ai, br, bi), 1e-5)
    assert _same_bits(old.real, new.real) and _same_bits(old.imag, new.imag)


@pytest.mark.parametrize("m,k,n", _GEMM_SHAPES)
def test_complex_gemm_bf16_tails(nki_backend, monkeypatch, m, k, n):
    from trnfft.nki.dispatch import complex_gemm_bf16

    ar, ai, br, bi = _gemm_inputs(m, k, n, seed=m + k + n, dtype=torch.bfloat16)
    old, new = _both_kernels(
        monkeypatch, lambda: complex_gemm_bf16(ComplexTensor(ar, ai), ComplexTensor(br, bi))
    )
    assert new.real.dtype == torch.float32
    # BF16 inputs are exact in FP64; only the FP32 PSUM accumulation rounds.
    _check_rows(new, _gemm_ref(ar, ai, br, bi), 1e-5)
    assert _same_bits(old.real, new.real) and _same_bits(old.imag, new.imag)


@pytest.mark.parametrize("precision", ["bf16", "ozaki"])
@pytest.mark.parametrize("rows,n", [(1000, 8), (4096, 16), (8192, 64)])
def test_bf16_packed_same_bits(nki_backend, monkeypatch, precision, rows, n):
    """The BF16 DFT-GEMM's row packing for n < 128 (trnfft#101 follow-up)
    gives the bits of the unpacked GEMM (``TRNFFT_GEMM_PACK=0``)."""
    xr, xi = _rand_pair(rows, n, seed=rows * n)
    drive = fc._fft_via_gemm_bf16 if precision == "bf16" else fc._fft_via_ozaki
    monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "1")
    monkeypatch.setenv("TRNFFT_GEMM_PACK", "0")
    old = drive(ComplexTensor(xr, xi), False)
    monkeypatch.setenv("TRNFFT_GEMM_PACK", "1")
    new = drive(ComplexTensor(xr, xi), False)
    _check_rows(new, _fft_ref(xr, xi), 2e-2 if precision == "bf16" else 1e-3)
    assert _same_bits(old.real, new.real) and _same_bits(old.imag, new.imag)


# (300, 1024, 1024): W^T is too large to keep resident in SBUF, so the linear
# kernel keeps the per-tile W transposes (``_HOIST_SBUF_BYTES``).
@pytest.mark.parametrize("m,k,n", _GEMM_SHAPES + [(300, 1024, 1024)])
def test_complex_linear_tails(nki_backend, monkeypatch, m, k, n):
    """``nn.ComplexLinear`` (``_complex_linear_kernel``): batch m, in k, out n."""
    from trnfft.nn import ComplexLinear

    torch.manual_seed(m + k + n)
    layer = ComplexLinear(k, n, bias=False)
    xr, xi = _rand_pair(m, k, seed=m)
    with torch.no_grad():
        old, new = _both_kernels(monkeypatch, lambda: layer(ComplexTensor(xr, xi)))
    wr, wi = layer.W_re.weight.detach(), layer.W_im.weight.detach()
    _check_rows(new, _gemm_ref(xr, xi, wr, wi, b_nk=True), 1e-5)
    assert _same_bits(old.real, new.real) and _same_bits(old.imag, new.imag)


@pytest.mark.parametrize("rows,cols", [(300, 256), (1000, 66), (129, 1024)])
def test_complex_mask_free_tail(nki_backend, rows, cols):
    """``_complex_mul_kernel`` with a (128, total/128) view whose free dim is
    over 512 and not a multiple of 512 (600, 515.6 -> falls back, 1032)."""
    from trnfft.nki.dispatch import complex_mask_apply

    mr, mi = _rand_pair(rows, cols, seed=rows)
    sr, si = _rand_pair(rows, cols, seed=cols)
    y = complex_mask_apply(ComplexTensor(mr, mi), ComplexTensor(sr, si))
    want = torch.complex(mr.double(), mi.double()) * torch.complex(sr.double(), si.double())
    _check_rows(y, want, 1e-6)
