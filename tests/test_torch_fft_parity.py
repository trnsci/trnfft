"""Argument parity with torch.fft: ``n`` / ``s``, ``dim`` and ``norm`` (#123).

Every public transform is compared with its ``torch.fft`` counterpart over
dim choices (negative, non-last, multiple), every norm mode, and n / s that
are shorter, longer and odd, for real and complex inputs at awkward
(non-power-of-2, prime) sizes. CPU (pytorch backend) only; the device check
for the NKI path is ``TestDeviceParity`` (``-m neuron``).
"""

import zlib

import pytest
import torch

import trnfft
from trnfft import ComplexTensor
from trnfft import api as trnfft_api

NORMS = [None, "backward", "ortho", "forward"]
SHAPE = (6, 5, 9)  # 6 and 9 are Bluestein sizes, 5 is prime


def _rand(shape, seed, complex_=False, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(*shape, generator=g, dtype=dtype)
    if not complex_:
        return re
    return torch.complex(re, torch.randn(*shape, generator=g, dtype=dtype))


def _seed(*key) -> int:
    """Deterministic per-case seed (str hash() is salted per process)."""
    return zlib.crc32(repr(key).encode()) % 10_000


def _as_torch(y):
    if isinstance(y, ComplexTensor):
        return torch.complex(y.real, y.imag)
    return y


def _trn_in(x):
    """torch complex -> ComplexTensor (trnfft's native complex type)."""
    return ComplexTensor(x) if torch.is_complex(x) else x


def _assert_matches(got, expected, rtol=1e-10, atol=1e-10):
    got = _as_torch(got)
    assert got.shape == expected.shape
    assert got.is_complex() == expected.is_complex()
    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)


# --- 1-D transforms -------------------------------------------------------

# (name, accepts complex input, accepts real input)
ONE_D = [
    ("fft", True, True),
    ("ifft", True, True),
    ("rfft", False, True),
    ("irfft", True, False),
    ("hfft", True, True),
    ("ihfft", False, True),
]
ONE_D_CASES = [
    (name, cplx) for name, c_ok, r_ok in ONE_D for cplx in (True, False) if (c_ok if cplx else r_ok)
]


@pytest.mark.parametrize("name, cplx", ONE_D_CASES)
@pytest.mark.parametrize("dim", [-1, 0, 1, -2, 2])
@pytest.mark.parametrize("n", [None, 3, 4, 7, 13, 16])
@pytest.mark.parametrize("norm", NORMS)
def test_1d_matches_torch(name, cplx, dim, n, norm):
    x = _rand(SHAPE, seed=_seed(name, dim), complex_=cplx)
    expected = getattr(torch.fft, name)(x, n=n, dim=dim, norm=norm)
    got = getattr(trnfft, name)(_trn_in(x), n=n, dim=dim, norm=norm)
    _assert_matches(got, expected)


@pytest.mark.parametrize("name", ["irfft", "hfft"])
@pytest.mark.parametrize("dim", [0, -2])
def test_1d_torch_complex_input(name, dim):
    """A torch complex tensor (not a ComplexTensor) is accepted directly."""
    x = _rand(SHAPE, seed=3, complex_=True)
    expected = getattr(torch.fft, name)(x, n=8, dim=dim, norm="ortho")
    got = getattr(trnfft, name)(x, n=8, dim=dim, norm="ortho")
    _assert_matches(got, expected)


@pytest.mark.parametrize("name", ["fft", "rfft", "ihfft", "ifft"])
@pytest.mark.parametrize("norm", NORMS)
def test_1d_float32(name, norm):
    x = _rand((4, 12, 3), seed=5, dtype=torch.float32)
    expected = getattr(torch.fft, name)(x, n=10, dim=1, norm=norm)
    got = getattr(trnfft, name)(x, n=10, dim=1, norm=norm)
    assert got.dtype == torch.float32
    _assert_matches(got, expected, rtol=1e-5, atol=1e-5)


def test_1d_on_1d_input():
    x = _rand((11,), seed=1, complex_=True)
    for norm in NORMS:
        _assert_matches(trnfft.fft(ComplexTensor(x), dim=0, norm=norm), torch.fft.fft(x, norm=norm))


def test_1d_integer_input_with_dim_and_norm():
    x = torch.arange(30).reshape(6, 5)
    expected = torch.fft.rfft(x, dim=0, norm="ortho")
    _assert_matches(trnfft.rfft(x, dim=0, norm="ortho"), expected.to(torch.complex64), 1e-5, 1e-5)


# --- 2-D and N-D transforms ------------------------------------------------

C2C_ND = ["fftn", "ifftn"]
R2C_ND = ["rfftn", "ihfftn"]  # real input only
C2R_ND = ["irfftn", "hfftn"]  # complex input (hfftn also takes real)

ND_DIMS = [None, (0,), (-1,), (1,), (-1, 0), (0, 2), (2, 0), (1, -1), (2, 0, 1), (-3, -2, -1)]


def _s_for(dims, pattern):
    """Sizes for ``dims``: shorter, longer, odd, or keep (-1) per position."""
    nd = len(SHAPE) if dims is None else len(dims)
    ds = range(-nd, 0) if dims is None else dims
    out = []
    for i, d in enumerate(ds):
        size = SHAPE[d]
        if pattern == "short":
            out.append(max(1, size - 2))
        elif pattern == "long":
            out.append(size + 3)
        elif pattern == "odd":
            out.append(7)
        else:  # "mixed": keep, longer, shorter, ... in turn
            out.append([-1, size + 3, max(1, size - 2)][i % 3])
    return tuple(out)


S_PATTERNS = [None, "short", "long", "odd", "mixed"]


def _nd_inputs(name):
    if name in R2C_ND:
        return [False]
    if name == "irfftn":
        return [True]
    return [True, False]


ND_CASES = [(n, c) for n in C2C_ND + R2C_ND + C2R_ND for c in _nd_inputs(n)]


@pytest.mark.parametrize("name, cplx", ND_CASES)
@pytest.mark.parametrize("dim", ND_DIMS)
@pytest.mark.parametrize("s_pattern", S_PATTERNS)
@pytest.mark.parametrize("norm", NORMS)
def test_nd_matches_torch(name, cplx, dim, s_pattern, norm):
    x = _rand(SHAPE, seed=_seed(name, dim), complex_=cplx)
    s = None if s_pattern is None else _s_for(dim, s_pattern)
    expected = getattr(torch.fft, name)(x, s=s, dim=dim, norm=norm)
    got = getattr(trnfft, name)(_trn_in(x), s=s, dim=dim, norm=norm)
    _assert_matches(got, expected)


@pytest.mark.parametrize("name", ["fftn", "ifftn", "rfftn", "irfftn", "hfftn", "ihfftn"])
@pytest.mark.parametrize("s", [(4,), (8, 3)])
def test_nd_s_without_dim_uses_last_dims(name, s):
    x = _rand(SHAPE, seed=9, complex_=name in ("irfftn", "hfftn"))
    expected = getattr(torch.fft, name)(x, s=s, norm="ortho")
    _assert_matches(getattr(trnfft, name)(_trn_in(x), s=s, norm="ortho"), expected)


TWO_D = ["fft2", "ifft2", "rfft2", "irfft2", "hfft2", "ihfft2"]
TWO_D_DIMS = [(-2, -1), (0, 2), (2, 0), (-1, 1), (0, 1)]


@pytest.mark.parametrize("name", TWO_D)
@pytest.mark.parametrize("dim", [*TWO_D_DIMS, "default"])
@pytest.mark.parametrize("s_pattern", S_PATTERNS)
@pytest.mark.parametrize("norm", NORMS)
def test_2d_matches_torch(name, dim, s_pattern, norm):
    cplx = name in ("fft2", "ifft2", "irfft2", "hfft2")
    x = _rand(SHAPE, seed=_seed(name, dim), complex_=cplx)
    kwargs = {"norm": norm}
    dims = (-2, -1) if dim == "default" else dim
    if dim != "default":
        kwargs["dim"] = dim
    if s_pattern is not None:
        kwargs["s"] = _s_for(dims, s_pattern)
    expected = getattr(torch.fft, name)(x, **kwargs)
    _assert_matches(getattr(trnfft, name)(_trn_in(x), **kwargs), expected)


@pytest.mark.parametrize("name", ["fftn", "rfftn", "ifft2", "irfftn"])
@pytest.mark.parametrize("norm", NORMS)
def test_nd_float32(name, norm):
    x = _rand((5, 8, 6), seed=2, complex_=name in ("ifft2", "irfftn"), dtype=torch.float32)
    expected = getattr(torch.fft, name)(x, dim=(0, 2), norm=norm)
    got = getattr(trnfft, name)(_trn_in(x), dim=(0, 2), norm=norm)
    _assert_matches(got, expected, rtol=1e-5, atol=1e-5)


def test_fftn_int_dim():
    """torch.fft.fftn accepts a bare int for ``dim``."""
    x = _rand(SHAPE, seed=4, complex_=True)
    _assert_matches(trnfft.fftn(ComplexTensor(x), dim=1), torch.fft.fftn(x, dim=1))


# --- Round trips under each norm --------------------------------------------


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize(
    "fwd, inv, real",
    [
        ("fft", "ifft", False),
        ("rfft", "irfft", True),
        ("ihfft", "hfft", True),
        ("fftn", "ifftn", False),
        ("rfftn", "irfftn", True),
        ("ihfftn", "hfftn", True),
    ],
)
def test_round_trip(fwd, inv, real, norm):
    x = _rand((6, 10), seed=7, complex_=not real)
    is_nd = fwd.endswith("n")
    kw = {"dim": (1, 0)} if is_nd else {"dim": 0}
    size = {"s": (10, 6)} if is_nd else {"n": 6}
    y = getattr(trnfft, fwd)(_trn_in(x), norm=norm, **kw)
    back = getattr(trnfft, inv)(y, norm=norm, **kw, **size)
    _assert_matches(back, x)


def test_ortho_preserves_energy():
    x = _rand((4, 13), seed=8, complex_=True)
    y = trnfft.fft(ComplexTensor(x), dim=1, norm="ortho")
    torch.testing.assert_close(_as_torch(y).abs().pow(2).sum(), x.abs().pow(2).sum())


# --- Defaults unchanged -----------------------------------------------------


@pytest.mark.parametrize("name", ["fft", "ifft", "hfft", "irfft"])
def test_default_norm_is_backward(name):
    x = ComplexTensor(_rand((3, 9), seed=6, complex_=True))
    a = _as_torch(getattr(trnfft, name)(x))
    b = _as_torch(getattr(trnfft, name)(x, norm="backward"))
    assert torch.equal(a, b)


# --- Kernel layout: dim is moved to the last axis on the host ---------------


@pytest.mark.parametrize(
    "call",
    [
        lambda x: trnfft.fft(x, dim=0, norm="ortho"),
        lambda x: trnfft.ifft(x, n=11, dim=-2),
        lambda x: trnfft.rfft(x.real, dim=1, norm="forward"),
        lambda x: trnfft.irfft(x, n=8, dim=0),
        lambda x: trnfft.fftn(x, dim=(0, 2), norm="ortho"),
        lambda x: trnfft.rfftn(x.real, dim=(2, 0)),
        lambda x: trnfft.irfftn(x, dim=(2, 1)),
    ],
)
def test_fft_core_sees_contiguous_last_axis(monkeypatch, call):
    """fft_core (and so every NKI kernel) always gets a contiguous (..., n) row
    layout transforming the last axis, whatever ``dim`` is: no kernel change."""
    seen = []
    real_core = trnfft_api.fft_core

    def spy(x, inverse=False, **kw):
        seen.append((x.real.is_contiguous(), x.imag.is_contiguous()))
        return real_core(x, inverse=inverse, **kw)

    monkeypatch.setattr(trnfft_api, "fft_core", spy)
    call(ComplexTensor(_rand(SHAPE, seed=10, complex_=True)))
    assert seen and all(a and b for a, b in seen)


# --- fftshift / ifftshift / fftfreq / rfftfreq ------------------------------


SHIFT_CASES = [
    ((7,), None),
    ((7,), 0),
    ((7,), -1),
    ((4, 8), None),
    ((4, 8), -1),
    ((4, 8), (-2,)),
    ((6, 5, 9), None),
    ((6, 5, 9), 0),
    ((6, 5, 9), (0, 2)),
    ((6, 5, 9), (-1, 1)),
]


@pytest.mark.parametrize("fn", ["fftshift", "ifftshift"])
@pytest.mark.parametrize("shape, dim", SHIFT_CASES)
def test_shift_matches_torch(fn, shape, dim):
    x = _rand(shape, seed=11, complex_=True)
    expected = getattr(torch.fft, fn)(x, dim=dim)
    # torch.Tensor in, torch.Tensor out
    got_t = getattr(trnfft, fn)(x.real, dim=dim)
    assert isinstance(got_t, torch.Tensor)
    assert torch.equal(got_t, expected.real)
    # ComplexTensor in, ComplexTensor out
    got_c = getattr(trnfft, fn)(ComplexTensor(x), dim=dim)
    assert isinstance(got_c, ComplexTensor)
    assert torch.equal(_as_torch(got_c), expected)


def test_shift_round_trip_and_copy():
    x = torch.arange(15.0).reshape(3, 5)
    assert torch.equal(trnfft.ifftshift(trnfft.fftshift(x)), x)
    y = trnfft.fftshift(x, dim=())
    assert torch.equal(y, x) and y.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("n", [1, 2, 7, 8, 13])
@pytest.mark.parametrize("d", [1.0, 0.25, 3.0])
@pytest.mark.parametrize("dtype", [None, torch.float64, torch.float32])
def test_freq_matches_torch(n, d, dtype):
    for name in ("fftfreq", "rfftfreq"):
        expected = getattr(torch.fft, name)(n, d, dtype=dtype)
        got = getattr(trnfft, name)(n, d, dtype=dtype)
        assert got.dtype == expected.dtype
        torch.testing.assert_close(got, expected)


def test_freq_requires_grad():
    assert trnfft.fftfreq(4, requires_grad=True).requires_grad
    assert not trnfft.rfftfreq(4).requires_grad


# --- Argument errors match torch.fft ----------------------------------------


@pytest.mark.parametrize("name", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"])
def test_bad_norm_raises(name):
    with pytest.raises(RuntimeError, match="Invalid normalization mode"):
        getattr(trnfft, name)(torch.randn(4, 8), norm="none")


@pytest.mark.parametrize("name", ["fftn", "rfftn", "irfftn", "fft2"])
def test_bad_norm_raises_nd(name):
    with pytest.raises(RuntimeError, match="Invalid normalization mode"):
        getattr(trnfft, name)(torch.randn(4, 8), norm="Ortho")


@pytest.mark.parametrize("dim", [2, -3])
def test_dim_out_of_range(dim):
    with pytest.raises(IndexError, match="Dimension out of range"):
        trnfft.fft(torch.randn(4, 8), dim=dim)
    with pytest.raises(IndexError, match="Dimension out of range"):
        trnfft.fftn(torch.randn(4, 8), dim=(0, dim))


@pytest.mark.parametrize("n", [0, -1, -5])
def test_invalid_n(n):
    """A non-positive n raises (n=-1 used to drop the last sample silently)."""
    for name in ("fft", "rfft", "irfft"):
        with pytest.raises(RuntimeError, match="Invalid number of data points"):
            getattr(trnfft, name)(torch.randn(4, 8), n=n)


def test_s_minus_one_keeps_size():
    """s[i] == -1 keeps the input size, as in torch.fft (it used to truncate by one)."""
    x = _rand((4, 6), seed=12, complex_=True)
    _assert_matches(trnfft.fftn(ComplexTensor(x), s=(-1, 9)), torch.fft.fftn(x, s=(-1, 9)))
    with pytest.raises(RuntimeError, match="Invalid number of data points"):
        trnfft.fftn(x, s=(-2, 9))


def test_s_dim_length_mismatch():
    with pytest.raises(RuntimeError, match="same length"):
        trnfft.fftn(torch.randn(4, 8), s=(4,), dim=(0, 1))
    with pytest.raises(RuntimeError, match="input tensor only has"):
        trnfft.fftn(torch.randn(4, 8), s=(4, 4, 4))


@pytest.mark.parametrize("name", ["rfft", "ihfft", "rfftn", "ihfftn", "rfft2"])
def test_real_only_rejects_complex(name):
    with pytest.raises(TypeError, match="expects a real input tensor"):
        getattr(trnfft, name)(ComplexTensor(torch.randn(4, 8), torch.randn(4, 8)))
    with pytest.raises(TypeError, match="expects a real input tensor"):
        getattr(trnfft, name)(torch.randn(4, 8, dtype=torch.complex64))


@pytest.mark.parametrize("name", ["rfftn", "irfftn", "hfftn", "ihfftn"])
def test_c2r_r2c_need_one_dim(name):
    with pytest.raises(RuntimeError, match="at least one axis"):
        getattr(trnfft, name)(torch.randn(4, 8), dim=())


# --- Device validation (manual, -m neuron) ----------------------------------


def _on_neuron() -> bool:
    from trnfft.nki.dispatch import HAS_NKI

    return HAS_NKI


@pytest.mark.neuron
@pytest.mark.skipif(not _on_neuron(), reason="requires Neuron hardware")
class TestDeviceParity:
    """The NKI path with non-last dim / norm: the kernels see the same
    last-axis layout, so this should match the CPU results above. Needs a
    hardware run to confirm (dispatch and device-resident transposes)."""

    @pytest.mark.parametrize(
        "name, kwargs",
        [
            ("fft", {"dim": 0, "norm": "ortho"}),
            ("ifft", {"dim": 1, "n": 128, "norm": "forward"}),
            ("rfft", {"dim": 0, "n": 512}),
            ("irfft", {"dim": 0, "n": 254, "norm": "ortho"}),
            ("fftn", {"dim": (0, 2), "norm": "ortho"}),
            ("fft2", {"dim": (2, 0), "norm": "forward"}),
            ("rfftn", {"dim": (2, 0)}),
            ("irfftn", {"dim": (0, 2), "norm": "ortho"}),
        ],
    )
    def test_matches_torch(self, nki_backend, name, kwargs):
        cplx = name in ("fft", "ifft", "irfft", "fftn", "fft2", "irfftn")
        x = _rand((128, 3, 64), seed=13, complex_=cplx, dtype=torch.float32)
        expected = getattr(torch.fft, name)(x, **kwargs)
        got = getattr(trnfft, name)(_trn_in(x), **kwargs)
        _assert_matches(got, expected, rtol=1e-4, atol=1e-4)
