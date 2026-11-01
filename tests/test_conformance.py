"""Reference-conformance suite (trnsci/trnsci#105).

Every public trnfft API is checked against an *independent* reference
(``torch.fft``, ``torch.stft``/``torch.istft``, dense complex torch ops and
``torch.nn.functional``), never backend against backend. Differential tests
cannot see a bug that every backend shares; three audit rounds (trnsci/trnsci#35)
found several of those here (irfft bins #104, integer inputs #113, stft batch
#114, ModReLU bias #115, ``dim=()`` aliasing #116).

Axes: sizes (1, 2, odd/prime, 127/129 around the tile, >512 Bluestein, 1000,
8192 six-step), dtypes (fp16/bf16/fp32/fp64, complex64/128, int, bool),
non-default kwargs (``n=``, ``s=``, ``dim=``, every STFT option, precision
modes), non-contiguous inputs, empty and size-1 shapes, fp64 ``gradcheck``,
input mutation / output aliasing, batch semantics and determinism.

Runs on CPU with the pytorch backend in the ``pytest -m "not neuron"`` CI job.
The same cases run on a Trainium box through the conftest options::

    pytest tests/test_conformance.py --backend nki --conformance-device neuron
    pytest tests/test_conformance.py --backend auto --conformance-device cpu

On a non-pytorch backend the tolerances are floored at FP32 level, and on a
non-CPU device the fp64 gradchecks are skipped (fp64 autograd on device is not
a trnfft contract).

Known open behaviour is ``xfail(strict=True)`` with its issue number, so the
case fails loudly (XPASS) once the behaviour changes and the marker must go:

- trnsci/trnfft#106 (needs decision): ``stft``/``istft(window=None)`` use
  Hann; ``torch.stft``/``torch.istft`` use ones.
- trnsci/trnfft#121: ``fftn``-family raise on a 0-size non-transformed dim.

``dim=``/``norm=`` (#123) are covered exhaustively by
``tests/test_torch_fft_parity.py``; this file runs a bounded cross-section of
them (every norm, a non-last dim, awkward sizes) so they also run on device.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import trnfft
from trnfft import ComplexTensor as CT
from trnfft.nn import ComplexBatchNorm1d, ComplexConv1d, ComplexLinear, ComplexModReLU

# ----------------------------------------------------------------- config
_STATE = {"backend": "pytorch", "device": torch.device("cpu")}

TOL = {
    torch.float64: 1e-9,
    torch.float32: 2e-4,
    torch.float16: 2e-2,
    torch.bfloat16: 6e-2,
}


@pytest.fixture(scope="module", autouse=True)
def _conformance_backend(request):
    backend = request.config.getoption("--backend")
    device = torch.device(request.config.getoption("--conformance-device"))
    old_backend, old_precision = trnfft.get_backend(), trnfft.get_precision()
    trnfft.set_backend(backend)
    _STATE["backend"], _STATE["device"] = backend, device
    if device.type != "cpu":
        torch.empty(1, device=device)  # fail fast without the device
    yield
    trnfft.set_backend(old_backend)
    trnfft.set_precision(old_precision)
    _STATE["backend"], _STATE["device"] = "pytorch", torch.device("cpu")


def on_reference_path() -> bool:
    return _STATE["backend"] == "pytorch" and _STATE["device"].type == "cpu"


def dev(t):
    d = _STATE["device"]
    if isinstance(t, CT):
        return CT(t.real.to(d), t.imag.to(d))
    if isinstance(t, torch.Tensor):
        return t.to(d)
    return t


def host(t):
    if isinstance(t, CT):
        return CT(t.real.detach().cpu(), t.imag.detach().cpu())
    return t.detach().cpu()


def c128(o):
    o = host(o)
    if isinstance(o, CT):
        return torch.complex(o.real.double(), o.imag.double())
    return o.to(torch.complex128)


def out_dtype(o):
    return o.real.dtype if isinstance(o, CT) else o.dtype


def check(ours, ref, dtype=None, tol=None):
    """Assert ``ours`` matches ``ref`` in shape, dtype and value (rel. max error)."""
    a, b = c128(ours), c128(ref)
    assert tuple(a.shape) == tuple(b.shape), f"shape {tuple(a.shape)} != ref {tuple(b.shape)}"
    if dtype is not None:
        assert out_dtype(ours) == dtype, f"dtype {out_dtype(ours)} != expected {dtype}"
    if a.numel() == 0:
        return
    assert torch.isfinite(a).all(), "non-finite output"
    t = tol if tol is not None else TOL.get(out_dtype(ours), 2e-4)
    if not on_reference_path():
        t = max(t, 2e-3)
    scale = b.abs().max().item() or 1.0
    rel = (a - b).abs().max().item() / scale
    assert rel <= t, f"rel err {rel:.3e} > {t:.0e}"


def gen(seed):
    return torch.Generator().manual_seed(seed)


def rand(kind, shape, dtype=torch.float64, seed=0):
    """kind 'r': real; 'c'/'h': complex of the matching precision."""
    if kind == "r":
        return torch.randn(*shape, generator=gen(seed), dtype=dtype)
    cd = torch.complex128 if dtype == torch.float64 else torch.complex64
    return torch.randn(*shape, generator=gen(seed), dtype=cd)


def ref_empty_safe(ref, x, **kw):
    """Reference for inputs with 0-size batch dims.

    oneMKL (Linux x86 torch wheels) rejects some 0-batch FFT plans, so run the
    reference with each 0-size dim set to 1 and slice the result back to 0.
    The transformed dims here are never 0-size.
    """
    zero = [d for d, n in enumerate(x.shape) if n == 0]
    if not zero:
        return ref(x, **kw)
    shape = [1 if n == 0 else n for n in x.shape]
    out = ref(torch.zeros(shape, dtype=x.dtype), **kw)
    for d in zero:
        out = out.narrow(d, 0, 0)
    return out


def need_cpu_autograd():
    if _STATE["device"].type != "cpu":
        pytest.skip("fp64 gradcheck is CPU-only")


# ------------------------------------------------------------ op tables
# name -> (trnfft call(x, **kw), torch reference(x, **kw), input kind)
#   kind "c": complex in, "r": real in, "h": one-sided (Hermitian) spectrum in
def _ct(x):
    return CT(x) if isinstance(x, torch.Tensor) and x.is_complex() else x


OPS_1D = {
    "fft": (lambda x, **k: trnfft.fft(_ct(x), **k), torch.fft.fft, "c"),
    "ifft": (lambda x, **k: trnfft.ifft(_ct(x), **k), torch.fft.ifft, "c"),
    "rfft": (lambda x, **k: trnfft.rfft(x, **k), torch.fft.rfft, "r"),
    "ihfft": (lambda x, **k: trnfft.ihfft(x, **k), torch.fft.ihfft, "r"),
    "irfft": (lambda x, **k: trnfft.irfft(CT(x), **k), torch.fft.irfft, "h"),
    "hfft": (lambda x, **k: trnfft.hfft(CT(x), **k), torch.fft.hfft, "h"),
}
OPS_ND = {
    "fft2": (lambda x, **k: trnfft.fft2(_ct(x), **k), torch.fft.fft2, "c"),
    "fftn": (lambda x, **k: trnfft.fftn(_ct(x), **k), torch.fft.fftn, "c"),
    "ifftn": (lambda x, **k: trnfft.ifftn(_ct(x), **k), torch.fft.ifftn, "c"),
    "rfft2": (lambda x, **k: trnfft.rfft2(x, **k), torch.fft.rfft2, "r"),
    "rfftn": (lambda x, **k: trnfft.rfftn(x, **k), torch.fft.rfftn, "r"),
    "irfft2": (lambda x, **k: trnfft.irfft2(CT(x), **k), torch.fft.irfft2, "h"),
    "irfftn": (lambda x, **k: trnfft.irfftn(CT(x), **k), torch.fft.irfftn, "h"),
    "ifft2": (lambda x, **k: trnfft.ifft2(_ct(x), **k), torch.fft.ifft2, "c"),
    "hfft2": (lambda x, **k: trnfft.hfft2(CT(x), **k), torch.fft.hfft2, "h"),
    "ihfft2": (lambda x, **k: trnfft.ihfft2(x, **k), torch.fft.ihfft2, "r"),
    "hfftn": (lambda x, **k: trnfft.hfftn(CT(x), **k), torch.fft.hfftn, "h"),
    "ihfftn": (lambda x, **k: trnfft.ihfftn(x, **k), torch.fft.ihfftn, "r"),
}
OPS_2D = ["fft2", "ifft2", "rfft2", "irfft2", "hfft2", "ihfft2"]
OPS_NDIM = ["fftn", "ifftn", "rfftn", "irfftn", "hfftn", "ihfftn"]

# Sizes: trivial, odd/prime, around the 128 tile, Bluestein (non-power-of-2
# and > 512), large non-power-of-2, and power-of-2 Cooley-Tukey.
SIZES = [1, 2, 3, 7, 16, 127, 128, 129, 513, 1000]


# ======================================================== 1-D transforms
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("op", list(OPS_1D))
def test_1d_sizes(op, n, dtype):
    f, ref, kind = OPS_1D[op]
    if kind == "h":
        # One-sided spectrum of an n-point real signal, with explicit n
        # (the irfft-bins class of bug, #104).
        x = rand("c", (3, n // 2 + 1), dtype, seed=n)
        check(f(dev(x), n=n), ref(x.to(torch.complex128), n=n), dtype=dtype)
    else:
        x = rand(kind, (3, n), dtype, seed=n)
        want = x.real.dtype if x.is_complex() else dtype
        check(f(dev(x)), ref(x.to(torch.complex128 if x.is_complex() else torch.float64)), want)


@pytest.mark.parametrize("n", [1, 2, 3, 9, 10, 11, 16, 33, 129])
@pytest.mark.parametrize("op", list(OPS_1D))
def test_1d_n_kwarg(op, n):
    """``n=`` both truncates and zero-pads (input length 10)."""
    f, ref, kind = OPS_1D[op]
    x = rand("r" if kind == "r" else "c", (2, 10), seed=n)
    check(f(dev(x), n=n), ref(x, n=n), dtype=torch.float64)


@pytest.mark.parametrize("op", ["irfft", "hfft"])
@pytest.mark.parametrize("m", [2, 3, 5, 9, 65])
def test_hermitian_default_n(op, m):
    """``n=None`` gives ``2 * (m - 1)`` output points."""
    f, ref, _ = OPS_1D[op]
    x = rand("c", (4, m), seed=m)
    check(f(dev(x)), ref(x))


@pytest.mark.parametrize("n", [8192])
def test_1d_large(n):
    """Six-step (N >= 8192, power of 2) and large Bluestein."""
    x = rand("c", (2, n), torch.float32, seed=n)
    check(trnfft.fft(dev(CT(x))), torch.fft.fft(x.to(torch.complex128)), tol=1e-3)
    check(trnfft.ifft(dev(CT(x))), torch.fft.ifft(x.to(torch.complex128)), tol=1e-3)


# ================================================================ dtypes
FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
INT_DTYPES = [torch.int16, torch.int32, torch.int64, torch.uint8]
REAL_IN_1D = ["fft", "ifft", "rfft", "ihfft"]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("n", [8, 12, 100])
@pytest.mark.parametrize("op", REAL_IN_1D)
def test_dtype_float_real_input(op, n, dtype):
    f, ref, _ = OPS_1D[op]
    x = (torch.randn(3, n, generator=gen(n)) * 5).to(dtype)
    check(f(dev(x)), ref(x.double()), dtype=dtype)


@pytest.mark.parametrize("dtype", INT_DTYPES, ids=str)
@pytest.mark.parametrize("op", REAL_IN_1D + ["fft2", "rfft2", "fftn", "rfftn"])
def test_dtype_int_input_promotes(op, dtype):
    """torch.fft promotes integer input to the default float dtype (#113)."""
    f, ref, _ = {**OPS_1D, **OPS_ND}[op]
    base = torch.randn(3, 12, generator=gen(3)) * 5
    x = base.abs().to(dtype) if dtype == torch.uint8 else base.to(dtype)
    check(f(dev(x)), ref(x.double()), dtype=torch.get_default_dtype())


@pytest.mark.parametrize("op", ["fft", "ifft", "rfft", "fft2"])
def test_dtype_bool_input(op):
    f, ref, _ = {**OPS_1D, **OPS_ND}[op]
    x = torch.randn(3, 12, generator=gen(4)) > 0
    check(f(dev(x)), ref(x.double()), dtype=torch.get_default_dtype())


@pytest.mark.parametrize(
    "cdtype,rdtype",
    [(torch.complex64, torch.float32), (torch.complex128, torch.float64)],
    ids=["c64", "c128"],
)
@pytest.mark.parametrize("op", ["fft", "ifft", "fft2", "fftn", "ifftn"])
@pytest.mark.parametrize("n", [12, 64, 129])
def test_dtype_complex_tensor_input(op, n, cdtype, rdtype):
    """A native torch complex tensor is accepted and keeps its precision."""
    f, ref, _ = {**OPS_1D, **OPS_ND}[op]
    z = torch.randn(3, n, dtype=cdtype, generator=gen(n))
    # Pass the torch complex tensor directly (not wrapped in ComplexTensor).
    fn = {"fft": trnfft.fft, "ifft": trnfft.ifft, "fft2": trnfft.fft2}.get(op)
    ours = fn(dev(z)) if fn is not None else f(dev(z))
    check(ours, ref(z.to(torch.complex128)), dtype=rdtype)


# ================================================== n-D transforms, s=/dim=
DIM_S = [
    (None, None),
    ((0, 2), None),
    ((2, 0), None),
    ((1,), (7,)),
    (None, (3, 8)),
    ((-1, -3), (5, 4)),
    ((0, 1, 2, 3), None),
    ((3, 1), (2, 9)),
    ((-1,), None),
    ((1, 2), (1, 1)),
]


@pytest.mark.parametrize("dim,s", DIM_S, ids=[f"dim={d}-s={s}" for d, s in DIM_S])
@pytest.mark.parametrize("op", OPS_NDIM)
def test_nd_dim_s(op, dim, s):
    f, ref, kind = OPS_ND[op]
    x4 = rand("r", (3, 4, 5, 6), seed=1)
    if kind == "r":
        x = x4
    elif kind == "c":
        x = rand("c", (3, 4, 5, 6), seed=2)
    else:
        x = torch.fft.rfftn(x4, dim=dim)
    check(f(dev(x), s=s, dim=dim), ref(x, s=s, dim=dim), dtype=torch.float64)


@pytest.mark.parametrize("s", [None, (3, 5), (8, 8), (2, 20), (1, 1), (127, 3)])
@pytest.mark.parametrize("op", OPS_2D)
def test_2d_s(op, s):
    f, ref, kind = OPS_ND[op]
    x4 = rand("r", (3, 4, 5, 6), seed=3)
    if kind == "r":
        x = x4
    elif kind == "c":
        x = rand("c", (3, 4, 5, 6), seed=4)
    else:
        x = torch.fft.rfft2(x4)
    check(f(dev(x), s=s), ref(x, s=s), dtype=torch.float64)


@pytest.mark.parametrize("shape", [(1, 1), (3, 7, 129), (2, 1000), (1, 513), (3, 1, 7)])
@pytest.mark.parametrize("op", list(OPS_ND))
def test_nd_shapes(op, shape):
    f, ref, kind = OPS_ND[op]
    x = rand("r" if kind == "r" else "c", shape, torch.float32, seed=sum(shape))
    if kind == "h" and shape[-1] < 2:
        pytest.skip("one-sided last dim needs >= 2 bins for the default s")
    check(f(dev(x)), ref(x.to(torch.complex128 if x.is_complex() else torch.float64)), tol=1e-3)


@pytest.mark.parametrize("op", ["fftn", "ifftn"])
def test_nd_empty_dim_is_copy(op):
    """``dim=()`` transforms nothing and returns a copy (#116)."""
    f, ref, _ = OPS_ND[op]
    z = rand("c", (3, 5), seed=5)
    a = dev(CT(z))
    out = f(a, dim=())
    check(out, ref(z, dim=()))
    out.real.add_(1.0)
    check(a, z)


# ================================================ non-contiguous layouts
def _views(base):
    return {
        "T": base.t(),
        "step": base[:, ::2],
        "step2": base[::2, 1::3],
        "expand": base[:1].expand(4, base.shape[1]),
        "as_strided": base.as_strided((5, 7), (3, 2)),
        "flip": base.flip(-1),
        "narrow": base[1:-1, 2:],
    }


VIEW_NAMES = list(_views(torch.zeros(8, 16)))


def _layout_input(kind, v):
    if kind == "c":
        return torch.complex(v, 0.5 * v.flip(0))
    if kind == "h":
        return torch.complex(v, 0.25 * v)
    return v


@pytest.mark.parametrize("view", VIEW_NAMES)
@pytest.mark.parametrize("n", [12])
@pytest.mark.parametrize("op", list(OPS_1D) + list(OPS_ND))
def test_layout_views(op, n, view):
    f, ref, kind = {**OPS_1D, **OPS_ND}[op]
    base = torch.randn(8, n, dtype=torch.float64, generator=gen(n))
    vin = _layout_input(kind, _views(base)[view])
    check(f(dev(vin)), ref(vin))


@pytest.mark.parametrize("view", VIEW_NAMES)
def test_layout_complextensor_mixed_strides(view):
    base = torch.randn(8, 16, dtype=torch.float64, generator=gen(6))
    v = _views(base)[view]
    im = torch.randn(v.shape[1], v.shape[0], dtype=torch.float64, generator=gen(7)).t()
    check(trnfft.fft(dev(CT(v, im))), torch.fft.fft(torch.complex(v, im)))


@pytest.mark.parametrize("op", ["fft", "ifft", "fft2"])
def test_layout_conj_view(op):
    f, ref, _ = {**OPS_1D, **OPS_ND}[op]
    z = torch.randn(4, 12, dtype=torch.complex128, generator=gen(8)).conj()
    assert z.is_conj()
    fn = {"fft": trnfft.fft, "ifft": trnfft.ifft, "fft2": trnfft.fft2}[op]
    check(fn(dev(z)), ref(z))


@pytest.mark.parametrize("op", ["fft2", "rfft2", "rfftn", "fftn"])
def test_layout_channels_last(op):
    f, ref, _ = OPS_ND[op]
    x = torch.randn(2, 3, 8, 8, dtype=torch.float64, generator=gen(9))
    x = x.to(memory_format=torch.channels_last)
    check(f(dev(x)), ref(x))


# ============================================================ edge shapes
EDGE_1D = [(0, 8), (0, 12), (2, 0, 8), (3, 1), (1,), (1, 1), (8,), (1, 2048), (1024, 2), (1, 3)]


@pytest.mark.parametrize("shape", EDGE_1D, ids=str)
@pytest.mark.parametrize("op", list(OPS_1D))
def test_edge_1d(op, shape):
    f, ref, kind = OPS_1D[op]
    if kind == "h" and shape[-1] < 2:
        pytest.skip("torch raises for a default n of 0")
    x = rand("r" if kind == "r" else "c", shape, seed=10)
    check(f(dev(x)), ref_empty_safe(ref, x))


EDGE_ND = [
    pytest.param((0, 4, 4), marks=pytest.mark.xfail(strict=True, reason="trnsci/trnfft#121")),
    (1, 1),
    (4, 1),
    (1, 4),
    (3, 5, 1),
]


@pytest.mark.parametrize("shape", EDGE_ND, ids=str)
@pytest.mark.parametrize("op", OPS_NDIM)
def test_edge_nd_all_dims(op, shape):
    f, ref, kind = OPS_ND[op]
    if kind == "h" and shape[-1] < 2:
        pytest.skip("torch raises for a default last size of 0")
    x = rand("r" if kind == "r" else "c", shape, seed=11)
    check(f(dev(x)), ref_empty_safe(ref, x))


@pytest.mark.parametrize("shape", [(0, 4, 4), (1, 1), (4, 1), (1, 4), (3, 5, 1)], ids=str)
@pytest.mark.parametrize("op", OPS_2D)
def test_edge_2d(op, shape):
    f, ref, kind = OPS_ND[op]
    if kind == "h" and shape[-1] < 2:
        pytest.skip("torch raises for a default last size of 0")
    x = rand("r" if kind == "r" else "c", shape, seed=12)
    check(f(dev(x)), ref_empty_safe(ref, x))


def test_edge_0d_raises_like_torch():
    s = torch.tensor(2.0, dtype=torch.float64)
    with pytest.raises((RuntimeError, IndexError)):
        torch.fft.fft(s)
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        trnfft.fft(dev(s))


# ================================================================== stft
def _torch_stft_batched(sig, **kw):
    """torch.stft takes 1-D or 2-D; fold extra leading dims."""
    flat = sig.reshape(-1, sig.shape[-1]) if sig.dim() > 2 else sig
    out = torch.stft(flat, return_complex=True, **kw)
    if sig.dim() > 2:
        out = out.reshape(*sig.shape[:-1], *out.shape[-2:])
    return out


STFT_KW = [
    dict(n_fft=16),
    dict(n_fft=16, hop_length=5, win_length=10),
    dict(n_fft=16, hop_length=1),
    dict(n_fft=16, normalized=True),
    dict(n_fft=15, hop_length=3),
    dict(n_fft=16, onesided=False),
    dict(n_fft=16, pad_mode="constant"),
    dict(n_fft=16, pad_mode="replicate"),
    dict(n_fft=16, pad_mode="circular"),
    dict(n_fft=16, center=False),
    dict(n_fft=16, center=False, hop_length=7, win_length=9),
]
STFT_SHAPES = [(64,), (1, 64), (3, 64), (2, 3, 64), (1, 1, 64)]


def _kw_id(kw):
    return ",".join(f"{k}={v}" for k, v in kw.items())


@pytest.mark.parametrize("shape", STFT_SHAPES, ids=str)
@pytest.mark.parametrize("kw", STFT_KW, ids=_kw_id)
def test_stft_vs_torch(kw, shape):
    sig = torch.randn(*shape, dtype=torch.float64, generator=gen(11))
    w = torch.hann_window(kw.get("win_length", kw["n_fft"]), dtype=torch.float64)
    ref = _torch_stft_batched(sig, window=w, **kw)
    check(trnfft.stft(dev(sig), window=dev(w), **kw), ref, dtype=torch.float64)


@pytest.mark.parametrize("shape", STFT_SHAPES, ids=str)
@pytest.mark.parametrize("kw", [k for k in STFT_KW if k.get("center", True)], ids=_kw_id)
def test_istft_vs_torch(kw, shape):
    ikw = {k: v for k, v in kw.items() if k != "pad_mode"}
    sig = torch.randn(*shape, dtype=torch.float64, generator=gen(12))
    w = torch.hann_window(kw.get("win_length", kw["n_fft"]), dtype=torch.float64)
    spec = _torch_stft_batched(sig, window=w, **kw)
    flat = spec.reshape(-1, *spec.shape[-2:]) if len(shape) > 2 else spec
    iref = torch.istft(flat, window=w, length=shape[-1], **ikw)
    if len(shape) > 2:
        iref = iref.reshape(*shape[:-1], -1)
    ours = trnfft.istft(dev(CT(spec)), window=dev(w), length=shape[-1], **ikw)
    check(ours, iref, dtype=torch.float64)
    # and the round trip recovers the signal
    check(ours, sig)


@pytest.mark.parametrize("L", [9, 16, 17, 1000, 4096])
def test_stft_signal_lengths(L):
    sig = torch.randn(L, dtype=torch.float64, generator=gen(L))
    w = torch.hann_window(16, dtype=torch.float64)
    check(
        trnfft.stft(dev(sig), 16, window=dev(w)),
        torch.stft(sig, 16, window=w, return_complex=True),
    )


@pytest.mark.parametrize("n_fft", [1, 2, 127, 129, 256])
def test_stft_n_fft_sizes(n_fft):
    sig = torch.randn(2, 600, dtype=torch.float64, generator=gen(n_fft))
    w = torch.hann_window(n_fft, dtype=torch.float64)
    hop = max(1, n_fft // 4)
    check(
        trnfft.stft(dev(sig), n_fft, hop_length=hop, window=dev(w)),
        torch.stft(sig, n_fft, hop_length=hop, window=w, return_complex=True),
    )


def test_stft_center_false_exact_frame():
    w = torch.hann_window(16, dtype=torch.float64)
    sig = torch.ones(16, dtype=torch.float64)
    check(
        trnfft.stft(dev(sig), 16, window=dev(w), center=False),
        torch.stft(sig, 16, window=w, center=False, return_complex=True),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)
def test_stft_dtype(dtype):
    x = torch.randn(2, 128, generator=gen(7)).to(dtype)
    w = torch.hann_window(32, dtype=dtype)
    check(
        trnfft.stft(dev(x), 32, hop_length=8, window=dev(w)),
        torch.stft(x.double(), 32, hop_length=8, window=w.double(), return_complex=True),
        dtype=dtype,
    )


def test_stft_noncontiguous_batch():
    sig = torch.randn(64, 3, dtype=torch.float64, generator=gen(13)).t()
    w = torch.hann_window(16, dtype=torch.float64)
    check(
        trnfft.stft(dev(sig), 16, window=dev(w)),
        torch.stft(sig, 16, window=w, return_complex=True),
    )


@pytest.mark.xfail(
    strict=True,
    reason="trnsci/trnfft#106 needs decision: stft(window=None) uses Hann, torch.stft uses ones",
)
def test_stft_default_window_matches_torch():
    sig = torch.randn(2, 256, dtype=torch.float64, generator=gen(14))
    check(trnfft.stft(dev(sig), 64), torch.stft(sig, 64, return_complex=True))


@pytest.mark.xfail(
    strict=True,
    reason="trnsci/trnfft#106 needs decision: istft(window=None) uses Hann, torch.istft uses ones",
)
def test_istft_default_window_matches_torch():
    sig = torch.randn(256, dtype=torch.float64, generator=gen(15))
    # torch.stft with a rectangular window and hop n_fft//4 satisfies NOLA.
    spec = torch.stft(sig, 64, return_complex=True)
    check(trnfft.istft(dev(CT(spec)), 64, length=256), torch.istft(spec, 64, length=256))


# ============================================================== autograd
def _split(r):
    return (r.real, r.imag) if isinstance(r, CT) else r


def _cplx_fns(n):
    return {
        "fft": lambda a, b: _split(trnfft.fft(CT(a, b))),
        "ifft": lambda a, b: _split(trnfft.ifft(CT(a, b))),
        "fft(n)": lambda a, b: _split(trnfft.fft(CT(a, b), n=n + 3)),
        "fft2": lambda a, b: _split(trnfft.fft2(CT(a, b))),
        "fftn(dim=0)": lambda a, b: _split(trnfft.fftn(CT(a, b), dim=(0,))),
        "ifftn": lambda a, b: _split(trnfft.ifftn(CT(a, b))),
        "ifft2(norm=ortho)": lambda a, b: _split(trnfft.ifft2(CT(a, b), norm="ortho")),
        "fft(dim=0,norm=forward)": lambda a, b: _split(trnfft.fft(CT(a, b), dim=0, norm="forward")),
        "fftshift": lambda a, b: _split(trnfft.fftshift(CT(a, b))),
        "complex_matmul": lambda a, b: _split(
            trnfft.complex_matmul(CT(a, b), CT(b.t().contiguous(), a.t().contiguous()))
        ),
    }


def _real_fns(n):
    h = n // 2 + 1

    def half(a):
        return CT(a[..., :h].contiguous(), (0.5 * a[..., :h]).contiguous())

    return {
        "rfft": lambda a: _split(trnfft.rfft(a)),
        "ihfft": lambda a: _split(trnfft.ihfft(a)),
        "rfft2": lambda a: _split(trnfft.rfft2(a)),
        "rfftn(dim=0)": lambda a: _split(trnfft.rfftn(a, dim=(0,))),
        "irfft": lambda a: _split(trnfft.irfft(half(a), n=n)),
        "hfft": lambda a: _split(trnfft.hfft(half(a), n=n)),
        "irfft2": lambda a: _split(trnfft.irfft2(half(a), s=(3, n))),
        "hfftn": lambda a: _split(trnfft.hfftn(half(a), s=(3, n))),
        "ihfftn": lambda a: _split(trnfft.ihfftn(a)),
        "ihfft2(norm=ortho)": lambda a: _split(trnfft.ihfft2(a, norm="ortho")),
        "rfft(dim=0)": lambda a: _split(trnfft.rfft(a, dim=0)),
        "stft": lambda a: _split(
            trnfft.stft(a.reshape(-1), n_fft=4, hop_length=2, window=torch.ones(4, dtype=a.dtype))
        ),
    }


AUTOGRAD_N = [6]
# Second order is linear for these ops; check it on one op per code path.
GRADGRAD = {"fft", "ifftn", "complex_matmul", "rfft", "irfft", "hfftn", "stft"}


@pytest.mark.parametrize("n", AUTOGRAD_N)
@pytest.mark.parametrize("name", list(_cplx_fns(8)))
def test_gradcheck_complex_in(name, n):
    need_cpu_autograd()
    f = _cplx_fns(n)[name]
    xr = torch.randn(2, n, dtype=torch.float64, generator=gen(n), requires_grad=True)
    xi = torch.randn(2, n, dtype=torch.float64, generator=gen(n + 1), requires_grad=True)
    if name == "complex_matmul":
        xr = xr[:, : n - 1].detach().requires_grad_()
        xi = xi[:, : n - 1].detach().requires_grad_()
    assert gradcheck(f, (xr, xi))
    if name in GRADGRAD:
        assert gradgradcheck(f, (xr, xi))


@pytest.mark.parametrize("n", AUTOGRAD_N)
@pytest.mark.parametrize("name", list(_real_fns(8)))
def test_gradcheck_real_in(name, n):
    need_cpu_autograd()
    f = _real_fns(n)[name]
    x = torch.randn(3, n, dtype=torch.float64, generator=gen(n + 2), requires_grad=True)
    assert gradcheck(f, (x,))
    if name in GRADGRAD:
        assert gradgradcheck(f, (x,))


def _grad_vs_torch(ours_fn, ref_fn, inp, out_shape):
    w = torch.randn(*out_shape, dtype=torch.complex128, generator=gen(3))
    a = inp.detach().clone().requires_grad_()
    (g1,) = torch.autograd.grad((c128_keep(ours_fn(a)) * w).real.sum(), a)
    b = inp.detach().clone().requires_grad_()
    (g2,) = torch.autograd.grad((ref_fn(b).to(torch.complex128) * w).real.sum(), b)
    check(g1, g2)


def c128_keep(o):
    if isinstance(o, CT):
        return torch.complex(o.real.double(), o.imag.double())
    return o.to(torch.complex128)


@pytest.mark.parametrize("n", [6, 8, 127])
@pytest.mark.parametrize(
    "name",
    ["rfft", "fft(real)", "fft2", "ihfft", "rfftn", "fft(n=+5)", "rfft(n=-2)"],
)
def test_grad_vs_torch_real_in(name, n):
    need_cpu_autograd()
    x = torch.randn(3, n, dtype=torch.float64, generator=gen(n + 3))
    h = n // 2 + 1
    ours, ref, shp = {
        "rfft": (trnfft.rfft, torch.fft.rfft, (3, h)),
        "fft(real)": (trnfft.fft, torch.fft.fft, (3, n)),
        "fft2": (trnfft.fft2, torch.fft.fft2, (3, n)),
        "ihfft": (trnfft.ihfft, torch.fft.ihfft, (3, h)),
        "rfftn": (trnfft.rfftn, torch.fft.rfftn, (3, h)),
        "fft(n=+5)": (
            lambda a: trnfft.fft(a, n=n + 5),
            lambda a: torch.fft.fft(a, n=n + 5),
            (3, n + 5),
        ),
        "rfft(n=-2)": (
            lambda a: trnfft.rfft(a, n=n - 2),
            lambda a: torch.fft.rfft(a, n=n - 2),
            (3, (n - 2) // 2 + 1),
        ),
    }[name]
    _grad_vs_torch(ours, ref, x, shp)


@pytest.mark.parametrize("n", [6, 7, 8, 127])
@pytest.mark.parametrize("op", ["irfft", "hfft"])
def test_grad_vs_torch_hermitian_in(op, n):
    need_cpu_autograd()
    ours_f = {"irfft": trnfft.irfft, "hfft": trnfft.hfft}[op]
    ref_f = {"irfft": torch.fft.irfft, "hfft": torch.fft.hfft}[op]
    z = torch.randn(3, n // 2 + 1, dtype=torch.complex128, generator=gen(n))
    w = torch.randn(3, n, dtype=torch.float64, generator=gen(5))
    zr = z.real.clone().requires_grad_()
    zi = z.imag.clone().requires_grad_()
    g1 = torch.autograd.grad((ours_f(CT(zr, zi), n=n) * w).sum(), (zr, zi))
    zt = z.clone().requires_grad_()
    (g2,) = torch.autograd.grad((ref_f(zt, n=n) * w).sum(), zt)
    check(torch.complex(g1[0], g1[1]), g2)


def test_grad_vs_torch_stft():
    need_cpu_autograd()
    sig = torch.randn(2, 64, dtype=torch.float64, generator=gen(16))
    win = torch.hann_window(16, dtype=torch.float64)
    ref_shape = torch.stft(sig, 16, hop_length=4, window=win, return_complex=True).shape
    _grad_vs_torch(
        lambda a: trnfft.stft(a, 16, hop_length=4, window=win),
        lambda a: torch.stft(a, 16, hop_length=4, window=win, return_complex=True),
        sig,
        ref_shape,
    )


# ================================================== aliasing and mutation
ALIAS_REAL = {
    "fft(real)": lambda a: trnfft.fft(a),
    "fft(CT)": lambda a: trnfft.fft(CT(a, a * 0 + 1)),
    "ifft": lambda a: trnfft.ifft(a),
    "rfft": lambda a: trnfft.rfft(a),
    "ihfft": lambda a: trnfft.ihfft(a),
    "fft2": lambda a: trnfft.fft2(a),
    "fftn": lambda a: trnfft.fftn(a),
    "fftn(dim=())": lambda a: trnfft.fftn(a, dim=()),
    "ifftn(dim=())": lambda a: trnfft.ifftn(a, dim=()),
    "fft(n=1 slice)": lambda a: trnfft.fft(a[:, :1]),
    "fft(n=same)": lambda a: trnfft.fft(a, n=a.shape[-1]),
    "rfft(n=same)": lambda a: trnfft.rfft(a, n=a.shape[-1]),
    "rfft2": lambda a: trnfft.rfft2(a),
    "rfftn": lambda a: trnfft.rfftn(a),
    "rfftn(dim=0)": lambda a: trnfft.rfftn(a, dim=(0,)),
    "stft": lambda a: trnfft.stft(
        a, 8, window=torch.hann_window(8, dtype=a.dtype, device=a.device)
    ),
}


@pytest.mark.parametrize("name", list(ALIAS_REAL))
def test_alias_real_input(name):
    a = dev(torch.randn(4, 16, dtype=torch.float64, generator=gen(17)))
    before = a.clone()
    out = ALIAS_REAL[name](a)
    assert torch.equal(host(a), host(before)), "input mutated"
    parts = (out.real, out.imag) if isinstance(out, CT) else (out,)
    for p in parts:
        assert p.untyped_storage().data_ptr() != a.untyped_storage().data_ptr(), (
            "output shares storage with input"
        )
        p.add_(1.0)
    assert torch.equal(host(a), host(before)), "writing the output mutated the input"


def _alias_spec():
    s = torch.stft(
        torch.randn(64, dtype=torch.float64, generator=gen(18)),
        8,
        window=torch.hann_window(8, dtype=torch.float64),
        return_complex=True,
    )
    return s


ALIAS_CT = {
    "irfft": lambda c: trnfft.irfft(c),
    "irfft(n=5)": lambda c: trnfft.irfft(c, n=5),
    "hfft": lambda c: trnfft.hfft(c),
    "irfft2": lambda c: trnfft.irfft2(c),
    "irfftn": lambda c: trnfft.irfftn(c),
    "fft(CT)": lambda c: trnfft.fft(c),
    "ifftn(CT)": lambda c: trnfft.ifftn(c),
    "fftn(CT, dim=())": lambda c: trnfft.fftn(c, dim=()),
    "complex_matmul": lambda c: trnfft.complex_matmul(
        c, CT(c.real.t().contiguous(), c.imag.t().contiguous())
    ),
    "istft": lambda c: trnfft.istft(
        c, 8, window=torch.hann_window(8, dtype=c.dtype, device=c.device)
    ),
}


@pytest.mark.parametrize("name", list(ALIAS_CT))
def test_alias_complex_input(name):
    z = _alias_spec() if name == "istft" else torch.randn(4, 16, dtype=torch.complex128)
    c = dev(CT(z))
    r0, i0 = c.real.clone(), c.imag.clone()
    out = ALIAS_CT[name](c)
    assert torch.equal(host(c.real), host(r0)) and torch.equal(host(c.imag), host(i0)), (
        "input mutated"
    )
    parts = (out.real, out.imag) if isinstance(out, CT) else (out,)
    for p in parts:
        for q in (c.real, c.imag):
            assert p.untyped_storage().data_ptr() != q.untyped_storage().data_ptr(), (
                "output shares storage with input"
            )


# ============================================================ batch dims
@pytest.mark.parametrize("bshape", [(), (2,), (2, 3), (1, 1, 2), (0,)], ids=str)
@pytest.mark.parametrize("op", list(OPS_1D))
def test_batch_dims_1d(op, bshape):
    f, ref, kind = OPS_1D[op]
    x = rand("r" if kind == "r" else "c", (*bshape, 12), seed=19)
    check(f(dev(x)), ref_empty_safe(ref, x))


def test_batch_split_vs_whole():
    for n in (64, 100, 1024):
        x = torch.randn(8, n, generator=gen(n))
        whole = host(trnfft.fft(dev(x)))
        parts = [host(trnfft.fft(dev(x[i : i + 3]))) for i in range(0, 8, 3)]
        check(
            CT(torch.cat([p.real for p in parts]), torch.cat([p.imag for p in parts])),
            whole.to_torch_complex(),
            tol=TOL[torch.float32],
        )


# ===================================================== dim= / norm= (#123)
NORMS = [None, "backward", "ortho", "forward"]


@pytest.mark.parametrize("norm", NORMS, ids=str)
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("op", list(OPS_1D))
def test_1d_dim_norm(op, dim, norm):
    """Every norm along last, first and middle dims, at awkward sizes."""
    f, ref, kind = OPS_1D[op]
    x = rand("r" if kind == "r" else "c", (33, 4, 127), seed=abs(dim) + 40)
    kw = dict(dim=dim, norm=norm)
    if kind == "h":
        kw["n"] = 2 * x.shape[dim] - 3  # odd n, fewer bins than the input
    check(f(dev(x), **kw), ref(x, **kw), dtype=torch.float64)


@pytest.mark.parametrize("norm", NORMS, ids=str)
@pytest.mark.parametrize("op", list(OPS_ND))
def test_nd_norm(op, norm):
    f, ref, kind = OPS_ND[op]
    x = rand("r" if kind == "r" else "c", (3, 7, 10), seed=41)
    dim = (0, 2) if op in OPS_2D else (2, 0)
    check(f(dev(x), dim=dim, norm=norm), ref(x, dim=dim, norm=norm), dtype=torch.float64)


@pytest.mark.parametrize("op", list(OPS_1D) + list(OPS_ND))
def test_norm_rejects_unknown(op):
    f, ref, kind = {**OPS_1D, **OPS_ND}[op]
    x = rand("r" if kind == "r" else "c", (4, 6), seed=42)
    with pytest.raises(RuntimeError):
        ref(x, norm="bogus")
    with pytest.raises((ValueError, RuntimeError)):
        f(dev(x), norm="bogus")


# ======================================================= shift / freq helpers
@pytest.mark.parametrize("dim", [None, 0, -1, (0, 2), (2, 0, 1)], ids=str)
@pytest.mark.parametrize("shape", [(1,), (7,), (4, 5, 6), (1, 127, 2)], ids=str)
@pytest.mark.parametrize("which", ["fftshift", "ifftshift"])
def test_shift(which, shape, dim):
    if dim is not None:
        dims = (dim,) if isinstance(dim, int) else dim
        if any(d >= len(shape) or d < -len(shape) for d in dims):
            pytest.skip("dim out of range for this rank")
    ours, ref = getattr(trnfft, which), getattr(torch.fft, which)
    x = torch.randn(*shape, dtype=torch.float64, generator=gen(43))
    z = torch.randn(*shape, dtype=torch.complex128, generator=gen(44))
    out = ours(dev(x), dim=dim)
    assert isinstance(out, torch.Tensor)
    check(out, ref(x, dim=dim), dtype=torch.float64)
    outc = ours(dev(CT(z)), dim=dim)
    assert isinstance(outc, CT)
    check(outc, ref(z, dim=dim), dtype=torch.float64)
    # fftshift and ifftshift invert each other on odd sizes too
    inv = getattr(trnfft, "ifftshift" if which == "fftshift" else "fftshift")
    check(inv(out, dim=dim), x)


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("d", [1.0, 0.1, 3])
@pytest.mark.parametrize("n", [1, 2, 7, 127, 128, 1000])
@pytest.mark.parametrize("which", ["fftfreq", "rfftfreq"])
def test_freq(which, n, d, dtype):
    ours = getattr(trnfft, which)(n, d, dtype=dtype, device=_STATE["device"])
    ref = getattr(torch.fft, which)(n, d, dtype=dtype)
    assert ours.dtype == ref.dtype and ours.shape == ref.shape
    assert ours.device.type == _STATE["device"].type
    torch.testing.assert_close(ours.cpu(), ref, rtol=1e-6, atol=0)


@pytest.mark.parametrize("which", ["fftfreq", "rfftfreq"])
def test_freq_requires_grad(which):
    out = getattr(trnfft, which)(8, requires_grad=True)
    assert out.requires_grad and out.is_leaf


# ========================================================= precision modes
@pytest.mark.parametrize("n", [1, 128, 129, 513])
@pytest.mark.parametrize("mode", ["fast", "kahan", "double"])
def test_precision_modes_vs_torch(mode, n):
    old = trnfft.get_precision()
    trnfft.set_precision(mode)
    try:
        x = torch.randn(4, n, generator=gen(n), dtype=torch.float32)
        check(trnfft.fft(dev(x)), torch.fft.fft(x.double()), dtype=torch.float32, tol=1e-3)
        check(trnfft.ifft(dev(x)), torch.fft.ifft(x.double()), dtype=torch.float32, tol=1e-3)
        xd = x.double()
        check(trnfft.fft(dev(xd)), torch.fft.fft(xd), dtype=torch.float64)
    finally:
        trnfft.set_precision(old)


@pytest.mark.parametrize("mode", ["bf16", "bf16_refined", "ozaki", "ozaki_hq"])
def test_precision_lowp_modes_bounded(mode):
    """The bf16-GEMM modes stay within bf16 accuracy and keep the dtype."""
    old = trnfft.get_precision()
    trnfft.set_precision(mode)
    try:
        x = torch.randn(4, 64, generator=gen(20), dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # ozaki fallback notice
            out = trnfft.fft(dev(x))
        check(out, torch.fft.fft(x.double()), dtype=torch.float32, tol=2e-2)
    finally:
        trnfft.set_precision(old)


def test_set_precision_rejects_unknown():
    with pytest.raises(ValueError):
        trnfft.set_precision("quad")
    assert trnfft.get_precision() in (
        "fast",
        "kahan",
        "double",
        "bf16",
        "bf16_refined",
        "ozaki",
        "ozaki_hq",
    )


# ============================================================ determinism
@pytest.mark.parametrize("n", [64, 100, 1024])
def test_determinism_repeat_and_plan_cache(n):
    x = torch.randn(8, n, generator=gen(n))
    a = host(trnfft.fft(dev(x)))
    b = host(trnfft.fft(dev(x)))
    assert torch.equal(a.real, b.real) and torch.equal(a.imag, b.imag), "repeat call differs"
    trnfft.clear_plan_cache()
    trnfft.ifft(dev(x))  # an inverse plan for n must not poison the forward one
    c = host(trnfft.fft(dev(x)))
    check(c, torch.fft.fft(x.double()), tol=1e-3)


# ================================================================== plans
@pytest.mark.parametrize("n", [1, 2, 3, 127, 128, 129, 1000, 1024])
@pytest.mark.parametrize("inverse", [False, True])
def test_create_plan(n, inverse):
    from trnfft.plan import FFTAlgorithm

    trnfft.clear_plan_cache()
    p = trnfft.create_plan(n, inverse=inverse)
    assert isinstance(p, trnfft.FFTPlan)
    assert p.n == n and p.inverse == inverse
    pow2 = n & (n - 1) == 0
    assert p.is_power_of_2 == pow2
    if pow2:
        assert p.algorithm == FFTAlgorithm.COOLEY_TUKEY
    else:
        assert p.algorithm == FFTAlgorithm.BLUESTEIN
        # Bluestein's linear convolution needs a power-of-2 length >= 2n - 1.
        assert p.padded_n >= 2 * n - 1 and p.padded_n & (p.padded_n - 1) == 0
    assert trnfft.create_plan(n, inverse=inverse) is p
    trnfft.clear_plan_cache()
    assert trnfft.create_plan(n, inverse=inverse) is not p


# ========================================================= ComplexTensor
def _z(shape, seed):
    return torch.randn(*shape, dtype=torch.complex128, generator=gen(seed))


def test_complextensor_construction():
    z = _z((3, 4), 21)
    check(CT(z), z, dtype=torch.float64)
    r = torch.randn(3, 4, dtype=torch.float64)
    check(CT(r), r.to(torch.complex128))
    check(CT(z.real, z.imag), z)
    check(CT(z).to_torch_complex(), z)
    assert CT(z).shape == z.shape and CT(z).dtype == torch.float64
    with pytest.raises(AssertionError):
        CT(torch.zeros(2), torch.zeros(3))
    with pytest.raises(AssertionError):
        CT(torch.zeros(2), torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize(
    "name",
    ["add", "sub", "mul", "add_scalar", "sub_scalar", "mul_scalar", "radd", "rmul",
     "add_real", "mul_real", "neg", "conj", "abs", "angle", "matmul"],
)  # fmt: skip
def test_complextensor_ops(name):
    a, b = _z((3, 4), 22), _z((3, 4), 23)
    r = torch.randn(3, 4, dtype=torch.float64, generator=gen(24))
    m = _z((4, 5), 25)
    A, B, M = dev(CT(a)), dev(CT(b)), dev(CT(m))
    R = dev(r)
    ours, ref = {
        "add": (lambda: A + B, lambda: a + b),
        "sub": (lambda: A - B, lambda: a - b),
        "mul": (lambda: A * B, lambda: a * b),
        "add_scalar": (lambda: A + 2.5, lambda: a + 2.5),
        "sub_scalar": (lambda: A - 2, lambda: a - 2),
        "mul_scalar": (lambda: A * -1.5, lambda: a * -1.5),
        "radd": (lambda: 2.5 + A, lambda: 2.5 + a),
        "rmul": (lambda: 3 * A, lambda: 3 * a),
        "add_real": (lambda: A + R, lambda: a + r),
        "mul_real": (lambda: A * R, lambda: a * r),
        "neg": (lambda: -A, lambda: -a),
        "conj": (lambda: A.conj(), lambda: a.conj().resolve_conj()),
        "abs": (lambda: A.abs(), lambda: a.abs()),
        "angle": (lambda: A.angle(), lambda: a.angle()),
        "matmul": (lambda: A @ M, lambda: a @ m),
    }[name]
    check(ours(), ref(), dtype=torch.float64)


def test_complextensor_from_polar():
    mag = torch.rand(3, 4, dtype=torch.float64, generator=gen(26))
    ph = torch.randn(3, 4, dtype=torch.float64, generator=gen(27))
    check(CT.from_polar(dev(mag), dev(ph)), torch.polar(mag, ph))


def test_complextensor_shape_ops():
    z = _z((2, 3, 4), 28)
    c = dev(CT(z))
    check(c[1], z[1])
    check(c[:, ::2, 1:], z[:, ::2, 1:])
    check(c.reshape(6, 4), z.reshape(6, 4))
    check(c.transpose(0, 2), z.transpose(0, 2))
    check(c.unsqueeze(1), z.unsqueeze(1))
    check(CT(z[:1]).squeeze(0), z[:1].squeeze(0))
    check(CT(z[:1]).squeeze(), z[:1].squeeze())
    d = c.clone()
    d.real.add_(1.0)
    check(c, z)  # clone does not alias
    e = dev(CT(z.clone()))
    e[0] = dev(CT(_z((3, 4), 29)))
    zz = z.clone()
    zz[0] = _z((3, 4), 29)
    check(e, zz)


# ========================================================= complex_matmul
MATMUL_SHAPES = [
    ((3, 4), (4, 5)),
    ((1, 1), (1, 1)),
    ((127, 129), (129, 3)),
    ((2, 3, 4), (4, 5)),
    ((2, 1, 3, 4), (5, 4, 2)),
    ((4,), (4, 5)),
    ((3, 4), (4,)),
    ((0, 3, 4), (4, 2)),
    ((3, 0), (0, 2)),
]


@pytest.mark.parametrize("sa,sb", MATMUL_SHAPES, ids=[f"{a}@{b}" for a, b in MATMUL_SHAPES])
@pytest.mark.parametrize("cdtype", [torch.complex64, torch.complex128], ids=["c64", "c128"])
def test_complex_matmul_vs_torch(sa, sb, cdtype):
    a = torch.randn(*sa, dtype=cdtype, generator=gen(30))
    b = torch.randn(*sb, dtype=cdtype, generator=gen(31))
    rdt = torch.float32 if cdtype == torch.complex64 else torch.float64
    ref = a.to(torch.complex128) @ b.to(torch.complex128)
    check(trnfft.complex_matmul(dev(CT(a)), dev(CT(b))), ref, dtype=rdt, tol=1e-5)


def test_complex_matmul_noncontiguous():
    a = torch.randn(7, 5, dtype=torch.complex128, generator=gen(32)).t()
    b = torch.randn(7, 4, dtype=torch.complex128, generator=gen(33))[:, ::2]
    check(trnfft.complex_matmul(dev(CT(a)), dev(CT(b))), a @ b)


# ============================================================== nn layers
def _seeded(layer):
    torch.manual_seed(0)
    for p in layer.parameters():
        with torch.no_grad():
            p.normal_()
    return layer.double()


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", [(4, 6), (2, 4, 6), (6,), (1, 6), (0, 6)], ids=str)
def test_complex_linear(shape, bias):
    L = _seeded(ComplexLinear(6, 3, bias=bias))
    W = torch.complex(L.W_re.weight, L.W_im.weight).detach()
    z = _z(shape, 34)
    ref = z @ W.t()
    if bias:
        ref = ref + torch.complex(L.bias_re, L.bias_im).detach()
    check(L.to(_STATE["device"])(dev(CT(z))), ref)


@pytest.mark.parametrize(
    "kw",
    [dict(), dict(padding=1), dict(stride=2), dict(stride=3, padding=2), dict(bias=False)],
    ids=_kw_id,
)
@pytest.mark.parametrize("length", [1, 10, 129])
def test_complex_conv1d(kw, length):
    C = _seeded(ComplexConv1d(3, 2, 3 if length >= 3 else 1, **kw))
    Wc = torch.complex(C.conv_re.weight, C.conv_im.weight).detach()
    z = _z((2, 3, length), 35)
    ref = torch.nn.functional.conv1d(
        z, Wc, stride=kw.get("stride", 1), padding=kw.get("padding", 0)
    )
    if kw.get("bias", True):
        ref = ref + torch.complex(C.bias_re, C.bias_im).detach()[:, None]
    check(C.to(_STATE["device"])(dev(CT(z))), ref)


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("shape", [(4, 3, 7), (5, 3)], ids=str)
def test_complex_batchnorm1d(shape, train):
    B = ComplexBatchNorm1d(3, eps=1e-3).double()
    ref_re = torch.nn.BatchNorm1d(3, eps=1e-3).double()
    ref_im = torch.nn.BatchNorm1d(3, eps=1e-3).double()
    z = _z(shape, 36)
    if not train:
        # one training step to move the running stats, then eval
        warm = _z(shape, 37)
        B.train()(CT(warm))
        ref_re.train()(warm.real)
        ref_im.train()(warm.imag)
        B.eval()
        ref_re.eval()
        ref_im.eval()
    B.to(_STATE["device"])
    out = B(dev(CT(z)))
    check(out, torch.complex(ref_re(z.real), ref_im(z.imag)).detach())


def _modrelu_ref(z, b):
    mag = z.abs()
    return torch.relu(mag + b) * z / mag.clamp_min(1e-8)


@pytest.mark.parametrize(
    "shape,bshape",
    [
        ((7, 5), (1, 5)),
        ((5,), (5,)),
        ((2, 5, 5), (1, 5, 1)),
        ((2, 5, 9), (1, 5, 1)),
        ((2, 5, 3, 4), (1, 5, 1, 1)),
        ((1, 5, 1), (1, 5, 1)),
    ],
    ids=str,
)
def test_complex_modrelu(shape, bshape):
    M = ComplexModReLU(5).double()
    with torch.no_grad():
        M.bias.copy_(torch.linspace(-1.0, 1.0, 5))
    z = _z(shape, 38)
    ref = _modrelu_ref(z, M.bias.detach().reshape(bshape))
    check(M.to(_STATE["device"])(dev(CT(z))), ref)


@pytest.mark.parametrize(
    "name", ["ComplexLinear", "ComplexConv1d", "ComplexModReLU", "ComplexBatchNorm1d"]
)
def test_nn_input_not_mutated(name):
    layer, shp = {
        "ComplexLinear": (ComplexLinear(16, 3).double(), (4, 16)),
        "ComplexConv1d": (ComplexConv1d(4, 2, 3).double(), (2, 4, 16)),
        "ComplexModReLU": (ComplexModReLU(4).double(), (2, 4, 16)),
        "ComplexBatchNorm1d": (ComplexBatchNorm1d(4).double(), (2, 4, 16)),
    }[name]
    c = dev(CT(_z(shp, 39)))
    r0, i0 = c.real.clone(), c.imag.clone()
    layer.to(_STATE["device"])(c)
    assert torch.equal(host(c.real), host(r0)) and torch.equal(host(c.imag), host(i0))


@pytest.mark.parametrize("name", ["ComplexLinear", "ComplexConv1d", "ComplexModReLU"])
def test_nn_gradcheck(name):
    need_cpu_autograd()
    torch.manual_seed(1)
    layer, shp = {
        "ComplexLinear": (ComplexLinear(4, 3).double(), (2, 4)),
        "ComplexConv1d": (ComplexConv1d(2, 2, 3, padding=1).double(), (2, 2, 6)),
        "ComplexModReLU": (ComplexModReLU(2).double(), (2, 2, 3)),
    }[name]
    xr = torch.randn(*shp, dtype=torch.float64, generator=gen(40), requires_grad=True)
    xi = torch.randn(*shp, dtype=torch.float64, generator=gen(41), requires_grad=True)
    assert gradcheck(lambda a, b: _split(layer(CT(a, b))), (xr, xi))
