"""No NKI kernel is reached with an operand dtype it does not take (trnsci#35).

float64 (and integer, bool, complex) tensors reached the FP32 kernels as is
and came back inf/NaN or garbage on Trn2 (#108, #131; the same class in the
other trnsci libraries). ``dispatch.KERNEL_DTYPES`` lists what each kernel
takes; ``_launch`` / ``_simulate_kernel`` / every ``nki_op`` body enforce it.
The public wrappers apply the suite float64 policy (trnfft#136) first:
backend "auto" computes float64 in torch at float64 accuracy (no kernel),
explicit "nki" runs the kernel on float32 copies with one ``UserWarning`` per
op, and ``TRNFFT_REQUIRE_NKI=1`` raises ``NotImplementedError``. Other
unsupported dtypes run in torch. Every result has the pytorch backend's dtype
(trnfft#137 item 2 for the low-precision modes).

These tests import ``trnfft.nki._native`` fresh under a stand-in
``torch_neuronx`` whose ``nki_op`` replaces every kernel op by a stub. The stub
records the operand dtypes, fails the test for any dtype outside ``ALLOWED``
(the table below, independent of the library's), and returns zero-filled
outputs of the op's ``fake_fn`` shape and dtype. The native dispatch path is
forced on CPU ("cpu" stands in for "neuron"); every public entry point that can
reach a kernel is driven with float64, bfloat16, float16, int32, int64, bool
and complex input under ``set_backend("nki")`` and ``"auto"``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
import warnings

import pytest
import torch

import trnfft
import trnfft.fft_core as fc
import trnfft.nki as trnfft_nki
import trnfft.nki.dispatch as dispatch
from trnfft import ComplexTensor, precision
from trnfft.nn import ComplexLinear

# The explicit-"nki" float64 warning (one per op; the fixture resets it per
# test). The tests that check it record warnings themselves.
# The device-resident float64 host-route warning (one per op) is ignored the
# same way; ``test_fp64_host_*`` record it themselves.
pytestmark = [
    pytest.mark.filterwarnings("ignore:trnfft .*float64 input:UserWarning"),
    pytest.mark.filterwarnings("ignore:trnfft .*float64 on the neuron device:UserWarning"),
]

_FP32 = {torch.float32}
_FP32_HALF = {torch.float32, torch.bfloat16, torch.float16}

# What each kernel takes. Deliberately not read from the library: a kernel
# added to `_native.OPS` without an entry here (or a library table that lets
# float64 through) fails `test_table_matches_ops`.
ALLOWED = {
    "complex_gemm": _FP32_HALF,
    "complex_gemm_t2d": _FP32_HALF,
    "complex_gemm_tn": _FP32,
    "complex_gemm_bf16": _FP32_HALF,
    "complex_gemm_bf16_t2d": _FP32_HALF,
    "complex_linear": _FP32_HALF,
    "complex_linear_t2d": _FP32_HALF,
    "complex_mul": _FP32_HALF,
    "butterfly_stage": _FP32_HALF,
    "butterfly_stage_kahan": _FP32_HALF,
    "stockham_radix4_stage": _FP32_HALF,
    "stockham_radix8_w8": _FP32_HALF,
    "stockham_radix8_w8_t2d": _FP32_HALF,
    "four_step_fft": _FP32,
}

# Where each kernel object lives (the drivers import them by name).
KERNEL_ATTRS = {
    "complex_gemm": ("dispatch", "_complex_gemm_kernel"),
    "complex_gemm_t2d": ("dispatch", "_complex_gemm_t2d_kernel"),
    "complex_gemm_tn": ("dispatch", "_complex_gemm_tn_kernel"),
    "complex_gemm_bf16": ("dispatch", "_complex_gemm_kernel_bf16"),
    "complex_gemm_bf16_t2d": ("dispatch", "_complex_gemm_bf16_t2d_kernel"),
    "complex_linear": ("dispatch", "_complex_linear_kernel"),
    "complex_linear_t2d": ("dispatch", "_complex_linear_t2d_kernel"),
    "complex_mul": ("dispatch", "_complex_mul_kernel"),
    "butterfly_stage": ("butterfly", "butterfly_stage_kernel"),
    "butterfly_stage_kahan": ("butterfly", "butterfly_stage_kernel_kahan"),
    "stockham_radix4_stage": ("stockham", "stockham_radix4_stage_kernel"),
    "stockham_radix8_w8": ("stockham", "stockham_radix8_w8_kernel"),
    "stockham_radix8_w8_t2d": ("stockham", "stockham_radix8_w8_t2d_kernel"),
    "four_step_fft": ("four_step", "four_step_fft_kernel"),
}

DTYPES = [
    torch.float64,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]
FLOAT_DTYPES = [torch.float64, torch.float32, torch.bfloat16, torch.float16]

# Ops the stubs saw over the whole module (checked by the last test).
SEEN: dict[str, set] = {}
VIOLATIONS: list[str] = []
# Kernel calls of the current test: (op, operand dtypes).
CALLS: list[tuple[str, set]] = []


class _StubOp:
    """Stand-in for a registered `nki_op`: dtype check + zero outputs."""

    def __init__(self, name, fn, fake_fn):
        self.name = name.split("::", 1)[1]
        self.fn = fn
        self.fake_fn = fake_fn

    def __call__(self, *args):
        dts = {a.dtype for a in args if isinstance(a, torch.Tensor)}
        SEEN.setdefault(self.name, set()).update(dts)
        CALLS.append((self.name, dts))
        bad = dts - ALLOWED.get(self.name, set())
        if bad:
            msg = f"{self.name} reached with {sorted(map(str, dts))}"
            VIOLATIONS.append(msg)
            raise AssertionError(msg)
        outs = self.fake_fn(*args)
        return tuple(torch.zeros(o.shape, dtype=o.dtype, device=o.device) for o in outs)

    def register_autograd(self, *a, **k):
        pass

    def register_fake(self, *a, **k):
        pass


class _Composite:
    """Stand-in for the `trnfft::fft` torch.library.custom_op (already
    registered process-wide by other tests): calls the host composite."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)

    def register_fake(self, f):
        return f

    def register_autograd(self, *a, **k):
        pass


def _fake_torch_neuronx():
    mod = types.ModuleType("torch_neuronx")
    real = sys.modules.get("torch_neuronx")
    if real is not None:
        # On a Neuron host the real runtime keeps reading module state from
        # `sys.modules["torch_neuronx"]` while the stand-in is installed (e.g.
        # `torch.manual_seed` -> `torch_neuronx._is_in_bad_fork`), so the
        # stand-in starts as a copy of the real module; `__spec__`, `nki_op`
        # and `device_count` are replaced below.
        mod.__dict__.update({k: v for k, v in vars(real).items() if not k.startswith("__")})
    mod.__spec__ = importlib.machinery.ModuleSpec("torch_neuronx", None)

    def nki_op(name, fn=None, mutates_args={}, fake_fn=None):  # noqa: B006
        def dec(f):
            return _StubOp(name, f, fake_fn)

        return dec if fn is None else dec(fn)

    mod.nki_op = nki_op
    mod.device_count = lambda: 1
    return mod


def _raw_kernel(*a, **k):
    raise AssertionError("raw @nki.jit kernel called on the native path")


@pytest.fixture
def native(monkeypatch):
    """`_native` imported fresh with stub ops; native dispatch forced on CPU."""
    monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_torch_neuronx())
    # Record the real state first, so teardown removes the stub-backed `_native`
    # imported below (a later test in the session would otherwise get stub ops).
    monkeypatch.setitem(sys.modules, "trnfft.nki._native", None)
    monkeypatch.delitem(sys.modules, "trnfft.nki._native", raising=False)
    had_attr = hasattr(trnfft_nki, "_native")
    old_attr = getattr(trnfft_nki, "_native", None)
    real_custom_op = torch.library.custom_op
    torch.library.custom_op = lambda name, mutates_args=(): _Composite
    try:
        mod = importlib.import_module("trnfft.nki._native")
    finally:
        torch.library.custom_op = real_custom_op
    monkeypatch.setitem(sys.modules, "trnfft.nki._native", mod)

    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_native_ops", mod.OPS)
    # The eager-handle fast path (#140) calls the real kernel through a cached
    # torch_neuronx handle and skips the op layer, so on a Neuron host it would
    # bypass the stub ops (and fail on the stand-in torch_neuronx module).
    monkeypatch.setattr(dispatch, "_EAGER_HANDLE_ENABLED", False)
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setattr(dispatch, "_USE_SIMULATOR", False)
    monkeypatch.setitem(sys.modules, "torch_xla", None)  # any import raises
    monkeypatch.setattr(precision, "_precision", "fast")
    monkeypatch.delenv("TRNFFT_REQUIRE_NKI", raising=False)
    monkeypatch.delenv("TRNFFT_PE_TRANSPOSE", raising=False)
    # Off-NKI the kernel objects are undefined; the drivers import them.
    from trnfft.nki import butterfly, four_step, stockham

    mods = {"dispatch": dispatch, "butterfly": butterfly, "stockham": stockham}
    mods["four_step"] = four_step
    for mod_name, attr in KERNEL_ATTRS.values():
        if getattr(mods[mod_name], attr, None) is None:
            # One distinct object per kernel (_simulate_kernel looks them up).
            def raw(*a, **k):
                _raw_kernel()

            monkeypatch.setattr(mods[mod_name], attr, raw, raising=False)
    monkeypatch.setattr(dispatch, "_backend", "nki")
    monkeypatch.setattr(dispatch, "_FP64_WARNED", set())
    monkeypatch.setattr(dispatch, "_FP64_HOST_WARNED", set())
    monkeypatch.delenv("TRNFFT_FP64_ON_DEVICE", raising=False)
    VIOLATIONS.clear()
    CALLS.clear()
    yield mod
    assert not VIOLATIONS, VIOLATIONS
    if had_attr:
        trnfft_nki._native = old_attr
    elif hasattr(trnfft_nki, "_native"):
        del trnfft_nki._native


@pytest.fixture(params=["nki", "auto"])
def backend(request, native, monkeypatch):
    monkeypatch.setattr(dispatch, "_backend", request.param)
    return request.param


def _pytorch(fn, *args, **kwargs):
    """`fn` under set_backend("pytorch") (the reference result)."""
    saved = dispatch._backend
    dispatch._backend = "pytorch"
    try:
        return fn(*args, **kwargs)
    finally:
        dispatch._backend = saved


def _rand(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    if not dtype.is_floating_point and not dtype.is_complex:
        return torch.randint(-5, 5, shape, generator=g).to(dtype)
    return torch.randn(shape, generator=g, dtype=torch.float64).to(dtype)


def _same_outcome(fn, *args, **kwargs):
    """Run `fn` on the NKI path and on the pytorch backend: both raise, or both
    return the same result dtype (and shape)."""
    try:
        ref = _pytorch(fn, *args, **kwargs)
    except (RuntimeError, TypeError, NotImplementedError):
        ref = None
    if ref is None:
        try:
            fn(*args, **kwargs)
        except (RuntimeError, TypeError, NotImplementedError):
            return None
        return None
    got = fn(*args, **kwargs)
    assert got.dtype == ref.dtype, (got.dtype, ref.dtype)
    assert tuple(got.shape) == tuple(ref.shape)
    return got


# ── The table ────────────────────────────────────────────────────────────────


def test_table_matches_ops(native):
    kernel_ops = set(native.OPS) - {"fft"}
    assert kernel_ops == set(ALLOWED), "a kernel op has no ALLOWED entry here"
    for op in kernel_ops:
        assert dispatch.KERNEL_DTYPES[op] == ALLOWED[op], op
        assert torch.float64 not in dispatch.KERNEL_DTYPES[op]
    assert dispatch._KERNEL_ATTRS == KERNEL_ATTRS


@pytest.mark.parametrize("op", sorted(ALLOWED))
def test_op_bodies_refuse_float64(native, op):
    """A direct `torch.ops.trnfft.<op>` call checks dtypes before the kernel."""
    stub = native.OPS[op]
    n_tensors = 7 if op == "four_step_fft" else 4
    args = [torch.ones(4, 4, dtype=torch.float64) for _ in range(n_tensors)]
    if op.startswith("butterfly"):
        args += [4, 0]
    with pytest.raises(TypeError, match=op):
        stub.fn(*args)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bool, torch.complex64])
def test_launch_refuses(native, dtype):
    t = torch.ones(4, 4, dtype=dtype)
    with pytest.raises(TypeError, match="complex_gemm"):
        dispatch._launch(_raw_kernel, "complex_gemm", t, t, t, t)
    with pytest.raises(TypeError, match="no KERNEL_DTYPES entry"):
        dispatch._launch(_raw_kernel, "not_a_kernel", t)


def test_xla_launch_refuses(native, monkeypatch):
    monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
    monkeypatch.setattr(dispatch, "_to_xla", lambda *t: (list(t), t[0].device))
    t = torch.ones(4, 4, dtype=torch.float64)
    with pytest.raises(TypeError, match="four_step_fft"):
        dispatch._launch(_raw_kernel, "four_step_fft", *([t] * 7))
    f = torch.ones(4, 4)
    out = dispatch._launch(lambda a, b, c, d: (a, b), "complex_mul", f, f, f, f)
    assert out[0].dtype == torch.float32


def test_simulate_kernel_refuses(native):
    t = torch.ones(4, 4, dtype=torch.float64)
    with pytest.raises(TypeError, match="complex_linear"):
        dispatch._simulate_kernel(dispatch._complex_linear_kernel, t, t, t, t, op="complex_linear")
    # Kernel looked up by identity when op= is omitted.
    with pytest.raises(TypeError, match="complex_mul"):
        dispatch._simulate_kernel(dispatch._complex_mul_kernel, t, t, t, t)
    with pytest.raises(TypeError, match="not a known kernel"):
        dispatch._simulate_kernel(lambda *a: a, torch.ones(2))


# ── Public FFT API ───────────────────────────────────────────────────────────

C2C = [trnfft.fft, trnfft.ifft, trnfft.hfft]
R2C = [trnfft.rfft, trnfft.ihfft]
ND = [trnfft.fft2, trnfft.ifft2, trnfft.fftn, trnfft.ifftn, trnfft.rfft2, trnfft.rfftn]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [8, 12, 64, 1024])
@pytest.mark.parametrize("fn", C2C + R2C, ids=lambda f: f.__name__)
def test_fft_1d(backend, fn, n, dtype):
    _same_outcome(fn, _rand((3, n), dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", ND, ids=lambda f: f.__name__)
def test_fft_nd(backend, fn, dtype):
    _same_outcome(fn, _rand((2, 8, 12), dtype))


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.int64])
@pytest.mark.parametrize("fn", [trnfft.irfft, trnfft.irfft2, trnfft.irfftn])
def test_c2r(backend, fn, dtype):
    re, im = _rand((3, 8, 9), dtype, 1), _rand((3, 8, 9), dtype, 2)
    _same_outcome(fn, ComplexTensor(re, im))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_torch_input(backend, dtype):
    x = _rand((4, 16), dtype)
    y = _same_outcome(trnfft.fft, x)
    assert y.dtype == x.real.dtype


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n", [2, 8, 16, 32, 64, 128, 256, 512, 2048])
@pytest.mark.parametrize("prec", ["fast", "kahan"])
@pytest.mark.parametrize("pe", ["1", "0"])
def test_stage_kernels(backend, monkeypatch, dtype, n, prec, pe):
    """GEMM paths off: butterfly / Stockham (radix-4, radix-8, mixed)."""
    monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 0)
    monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", pe)
    x = ComplexTensor(_rand((2, n), dtype, 1), _rand((2, n), dtype, 2))
    monkeypatch.setattr(precision, "_precision", prec)
    _same_outcome(trnfft.fft, x)
    _same_outcome(trnfft.ifft, x)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.int32])
@pytest.mark.parametrize("n", [8, 100, 256, 1024, 2048, 8192])
@pytest.mark.parametrize(
    "prec", ["fast", "kahan", "double", "bf16", "bf16_refined", "ozaki", "ozaki_hq"]
)
def test_precision_modes(backend, monkeypatch, dtype, n, prec):
    """Every precision mode, DFT-GEMM, four-step (fused and two-GEMM),
    six-step and Bluestein sizes."""
    if n == 8192 and prec not in ("fast", "kahan"):
        pytest.skip("six-step is fast/kahan only")
    monkeypatch.setattr(precision, "_precision", prec)
    x = _rand((2, n), dtype)
    # Half input in the BF16-GEMM modes returns the input dtype, as the
    # pytorch backend does (trnfft#137 item 2).
    _same_outcome(trnfft.fft, x)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
@pytest.mark.parametrize("pe", ["1", "0"])
def test_four_step_unfused_and_t2d(backend, monkeypatch, dtype, pe):
    monkeypatch.setattr(fc, "_FOUR_STEP_BATCH_FUSED", False)
    monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", pe)
    _same_outcome(trnfft.fft, _rand((3, 2048), dtype))
    _same_outcome(trnfft.fft, _rand((3, 64), dtype))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.int64, torch.bool])
def test_stft_istft(backend, dtype):
    x = _rand((2, 256), dtype)
    if not (dtype.is_floating_point):
        with pytest.raises((RuntimeError, TypeError)):
            trnfft.stft(x, 64)
        return
    spec = _same_outcome(trnfft.stft, x, 64)
    _same_outcome(trnfft.istft, spec, 64, length=256)


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool])
def test_fft_core_refuses_integer_parts(native, dtype):
    x = ComplexTensor(_rand((2, 8), dtype), _rand((2, 8), dtype))
    with pytest.raises(TypeError, match="floating-point"):
        fc.fft_core(x)


# ── nn layers and the dispatch-level complex ops ─────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(4, 6), (130, 6), (2, 3, 6)])
@pytest.mark.parametrize("pe", ["1", "0"])
def test_complex_linear_layer(backend, monkeypatch, dtype, shape, pe):
    monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", pe)
    torch.manual_seed(0)
    layer = ComplexLinear(6, 3).to(dtype)
    x = ComplexTensor(_rand(shape, dtype, 1), _rand(shape, dtype, 2))
    y = layer(x)
    ref = _pytorch(layer, x)
    assert y.real.dtype == ref.real.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_complex_linear_integer_input_runs_in_torch(backend, dtype):
    torch.manual_seed(0)
    layer = ComplexLinear(6, 3)
    x = ComplexTensor(_rand((4, 6), dtype, 1), _rand((4, 6), dtype, 2))
    with pytest.raises(RuntimeError):  # torch: mixed int/float matmul
        _pytorch(layer, x)
    with pytest.raises(RuntimeError):
        layer(x)
    assert "complex_linear" not in SEEN or not (SEEN["complex_linear"] - _FP32_HALF)


def test_complex_linear_float64(backend):
    """#131 / #132 / #136: float64 returns float64; "nki" runs the fp32
    kernel, "auto" computes in torch."""
    layer = ComplexLinear(6, 3).double()
    x = ComplexTensor(_rand((4, 6), torch.float64, 1), _rand((4, 6), torch.float64, 2))
    assert layer(x).real.dtype == torch.float64
    if backend == "nki":
        assert [op for op, _ in CALLS] == ["complex_linear"]
        assert CALLS[0][1] == {torch.float32}
    else:
        assert CALLS == []


def _ct(shape, dtype, seed=0):
    return ComplexTensor(_rand(shape, dtype, seed), _rand(shape, dtype, seed + 1))


@pytest.mark.parametrize("dtype", DTYPES[:6])
@pytest.mark.parametrize("pe", ["1", "0"])
def test_complex_gemm(backend, monkeypatch, dtype, pe):
    monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", pe)
    a, b = _ct((4, 6), dtype), _ct((6, 5), dtype, 2)
    _same_outcome(lambda: dispatch.complex_gemm(a, b))


@pytest.mark.parametrize("dtype", DTYPES[:6])
@pytest.mark.parametrize("shape", [(128,), (2, 128), (5, 7)])
def test_complex_mask_apply(backend, dtype, shape):
    m, s = _ct(shape, dtype), _ct(shape, dtype, 2)
    _same_outcome(lambda: dispatch.complex_mask_apply(m, s))


def test_complex_mask_apply_float64_odd_size_stays_float64(backend):
    """Sizes the kernel cannot take keep their full-precision torch route."""
    m, s = _ct((5, 7), torch.float64), _ct((5, 7), torch.float64, 2)
    got = dispatch.complex_mask_apply(m, s)
    ref = m * s
    assert torch.equal(got.real, ref.real) and torch.equal(got.imag, ref.imag)


@pytest.mark.parametrize("dtype", DTYPES[:6])
@pytest.mark.parametrize("pe", ["1", "0"])
def test_complex_gemm_bf16(backend, monkeypatch, dtype, pe):
    monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", pe)
    a, b = _ct((4, 6), dtype), _ct((6, 5), dtype, 2)
    got = _same_outcome(lambda: dispatch.complex_gemm_bf16(a, b))
    assert got is None or got.dtype == torch.float32


@pytest.mark.parametrize("dtype", DTYPES[:6])
def test_complex_linear_direct(backend, dtype):
    x = _ct((4, 6), dtype)
    w = _ct((3, 6), dtype, 2)
    _same_outcome(lambda: dispatch.complex_linear(x, w.real, w.imag))


def test_mixed_float32_float64_linear(backend):
    """fp32 activations with fp64 weights (#132's promote_types case)."""
    x = _ct((4, 6), torch.float32)
    w = _ct((3, 6), torch.float64, 2)
    y = dispatch.complex_linear(x, w.real, w.imag)
    assert y.real.dtype == torch.float64
    if backend == "auto":
        # torch at float64 on the promoted operands
        xd = ComplexTensor(x.real.double(), x.imag.double())
        want = _pytorch(dispatch.complex_linear, xd, w.real, w.imag)
        assert torch.equal(y.real, want.real) and torch.equal(y.imag, want.imag)
        assert CALLS == []
    else:
        assert CALLS and all(d == {torch.float32} for _, d in CALLS)


# ── The float64 policy (trnfft#136): auto -> torch, nki -> cast + warn ───────


def _fft_ref(x, inverse=False):
    z = torch.complex(x.real, x.imag)
    return torch.fft.ifft(z) if inverse else torch.fft.fft(z)


def _x64(shape, seed=0):
    return _ct(shape, torch.float64, seed)


# (op name in the warning, call on float64 operands)
FP64_CALLS = {
    "fft-gemm": ("fft", lambda: trnfft.fft(_x64((3, 64)))),
    "fft-bluestein": ("fft", lambda: trnfft.fft(_x64((3, 12)))),
    "fft-four-step": ("fft", lambda: trnfft.fft(_x64((2, 2048)))),
    "ifft": ("fft", lambda: trnfft.ifft(_x64((3, 64)))),
    "rfft": ("fft", lambda: trnfft.rfft(_rand((3, 64), torch.float64))),
    "complex128": ("fft", lambda: trnfft.fft(_rand((3, 64), torch.complex128))),
    "complex_gemm": (
        "complex_gemm",
        lambda: dispatch.complex_gemm(_x64((4, 6)), _x64((6, 5), 2)),
    ),
    "complex_linear": (
        "complex_linear",
        lambda: dispatch.complex_linear(
            _x64((4, 6)), _rand((3, 6), torch.float64, 2), _rand((3, 6), torch.float64, 3)
        ),
    ),
    "ComplexLinear": (
        "complex_linear",
        lambda: _seeded_layer().double()(_x64((4, 6))),
    ),
    "complex_mask_apply": (
        "complex_mask_apply",
        lambda: dispatch.complex_mask_apply(_x64((2, 128)), _x64((2, 128), 2)),
    ),
    "complex_gemm_bf16": (
        "complex_gemm_bf16",
        lambda: dispatch.complex_gemm_bf16(_x64((4, 6)), _x64((6, 5), 2)),
    ),
}


def _seeded_layer():
    torch.manual_seed(0)
    return ComplexLinear(6, 3)


def _fp64_warnings(rec):
    return [
        w for w in rec if issubclass(w.category, UserWarning) and "float64 input" in str(w.message)
    ]


@pytest.mark.parametrize("case", sorted(FP64_CALLS))
def test_float64_auto_computes_in_torch(native, monkeypatch, case):
    """backend "auto": float64 never reaches a kernel, the result is the
    pytorch backend's (bit for bit) and float64-accurate, and no warning."""
    monkeypatch.setattr(dispatch, "_backend", "auto")
    _, call = FP64_CALLS[case]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = call()
    assert CALLS == [], CALLS
    assert not _fp64_warnings(rec)
    ref = _pytorch(call)
    assert got.real.dtype == ref.real.dtype
    assert torch.equal(got.real, ref.real) and torch.equal(got.imag, ref.imag)
    if case.startswith("fft"):
        x = _x64({"fft-gemm": (3, 64), "fft-bluestein": (3, 12), "fft-four-step": (2, 2048)}[case])
        want = _fft_ref(x)
        err = (torch.complex(got.real, got.imag) - want).abs().max() / want.abs().max()
        assert err < 1e-12, err


@pytest.mark.parametrize("case", sorted(FP64_CALLS))
def test_float64_nki_casts_and_warns_once(native, case):
    """backend "nki": the kernel gets float32 operands, the result has the
    pytorch backend's dtype, and exactly one UserWarning per op over repeated
    calls."""
    op, call = FP64_CALLS[case]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        outs = [call() for _ in range(3)]
    assert CALLS, "no kernel was called"
    assert all(d <= {torch.float32} for _, d in CALLS), CALLS
    ws = _fp64_warnings(rec)
    assert len(ws) == 1, [str(w.message) for w in ws]
    assert f"trnfft {op}:" in str(ws[0].message)
    ref = _pytorch(call)
    assert all(o.real.dtype == ref.real.dtype for o in outs)


def test_float64_nki_warns_once_per_op(native):
    """Each op warns once; a second op still gets its own warning."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(2):
            for case in ("fft-gemm", "ifft", "complex_gemm", "complex_mask_apply"):
                FP64_CALLS[case][1]()
    ops = sorted(str(w.message).split(":")[0] for w in _fp64_warnings(rec))
    assert ops == ["trnfft complex_gemm", "trnfft complex_mask_apply", "trnfft fft"]


def test_float64_double_mode_unchanged(backend, monkeypatch):
    """precision="double" keeps its host FP64 route on both backends: no
    kernel, no warning, float64 accuracy."""
    monkeypatch.setattr(precision, "_precision", "double")
    for shape in ((3, 64), (3, 12), (2, 2048)):
        x = _x64(shape)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got = trnfft.fft(x)
        assert not _fp64_warnings(rec)
        want = _fft_ref(x)
        err = (torch.complex(got.real, got.imag) - want).abs().max() / want.abs().max()
        assert err < 1e-12, (shape, err)
    assert CALLS == []


# ── float64 already on the neuron device: computed on the host ──────────────
#
# Device aten computes float64 at FP32 precision (measured on Trn2), so under
# "auto" a float64 / complex128 operand on the neuron device is copied to the
# host, computed there in float64 and returned to its device
# (``dispatch._fp64_host_guard``). "cpu" stands in for "neuron" (the `native`
# fixture), so the host copy and the return copy are spied on to see the route.


def _c(z):
    """A ComplexTensor or torch tensor as one torch tensor (complex or real)."""
    return torch.complex(z.real, z.imag) if isinstance(z, ComplexTensor) else z


def _relerr(got, want):
    got, want = _c(got), _c(want)
    return ((got - want).abs().max() / want.abs().max()).item()


def _z(shape, seed=0):
    return _rand(shape, torch.complex128, seed)


def _r(shape, seed=0):
    return _rand(shape, torch.float64, seed)


def _stft_ref(x):
    with dispatch._torch_route():
        return _pytorch(trnfft.stft, x, 64)


# op name in the warning -> (call on float64 / complex128 operands, fp64 CPU reference)
FP64_HOST_CALLS = {
    "fft": (lambda: trnfft.fft(_z((3, 64))), lambda: torch.fft.fft(_z((3, 64)))),
    "ifft": (lambda: trnfft.ifft(_x64((3, 12))), lambda: torch.fft.ifft(_c(_x64((3, 12))))),
    "rfft": (lambda: trnfft.rfft(_r((3, 64))), lambda: torch.fft.rfft(_r((3, 64)))),
    "irfft": (lambda: trnfft.irfft(_z((3, 33))), lambda: torch.fft.irfft(_z((3, 33)))),
    "hfft": (lambda: trnfft.hfft(_z((3, 33))), lambda: torch.fft.hfft(_z((3, 33)))),
    "ihfft": (lambda: trnfft.ihfft(_r((3, 64))), lambda: torch.fft.ihfft(_r((3, 64)))),
    "fft2": (lambda: trnfft.fft2(_z((2, 8, 16))), lambda: torch.fft.fft2(_z((2, 8, 16)))),
    "ifft2": (lambda: trnfft.ifft2(_z((2, 8, 16))), lambda: torch.fft.ifft2(_z((2, 8, 16)))),
    "fftn": (lambda: trnfft.fftn(_z((4, 8, 16))), lambda: torch.fft.fftn(_z((4, 8, 16)))),
    "ifftn": (lambda: trnfft.ifftn(_z((4, 8, 16))), lambda: torch.fft.ifftn(_z((4, 8, 16)))),
    "rfft2": (lambda: trnfft.rfft2(_r((2, 8, 16))), lambda: torch.fft.rfft2(_r((2, 8, 16)))),
    "irfft2": (lambda: trnfft.irfft2(_z((2, 8, 9))), lambda: torch.fft.irfft2(_z((2, 8, 9)))),
    "rfftn": (lambda: trnfft.rfftn(_r((4, 8, 16))), lambda: torch.fft.rfftn(_r((4, 8, 16)))),
    "irfftn": (lambda: trnfft.irfftn(_z((4, 8, 9))), lambda: torch.fft.irfftn(_z((4, 8, 9)))),
    "hfft2": (lambda: trnfft.hfft2(_z((2, 8, 9))), lambda: torch.fft.hfft2(_z((2, 8, 9)))),
    "ihfft2": (lambda: trnfft.ihfft2(_r((2, 8, 16))), lambda: torch.fft.ihfft2(_r((2, 8, 16)))),
    "hfftn": (lambda: trnfft.hfftn(_z((4, 8, 9))), lambda: torch.fft.hfftn(_z((4, 8, 9)))),
    "ihfftn": (lambda: trnfft.ihfftn(_r((4, 8, 16))), lambda: torch.fft.ihfftn(_r((4, 8, 16)))),
    "stft": (lambda: trnfft.stft(_r((2, 256)), 64), lambda: _stft_ref(_r((2, 256)))),
    # istft(stft(x)) == x: the reference is the signal itself.
    "istft": (lambda: trnfft.istft(_stft_ref(_r((2, 256))), 64, length=256), lambda: _r((2, 256))),
    "complex_gemm": (
        lambda: dispatch.complex_gemm(_x64((4, 6)), _x64((6, 5), 2)),
        lambda: _c(_x64((4, 6))) @ _c(_x64((6, 5), 2)),
    ),
    "complex_linear": (
        lambda: dispatch.complex_linear(_x64((4, 6)), _r((3, 6), 2), _r((3, 6), 3)),
        lambda: _c(_x64((4, 6))) @ torch.complex(_r((3, 6), 2), _r((3, 6), 3)).T,
    ),
    "complex_mask_apply": (
        lambda: dispatch.complex_mask_apply(_x64((2, 128)), _x64((2, 128), 2)),
        lambda: _c(_x64((2, 128))) * _c(_x64((2, 128), 2)),
    ),
    "ComplexLinear": (
        lambda: _seeded_layer().double()(_x64((4, 6))),
        lambda: _layer_ref(_seeded_layer().double(), _x64((4, 6))),
    ),
}


def _layer_ref(layer, x):
    w = torch.complex(layer.W_re.weight, layer.W_im.weight).detach()
    b = torch.complex(layer.bias_re, layer.bias_im).detach()
    return _c(x) @ w.T + b


@pytest.fixture
def host_spy(native, monkeypatch):
    """backend "auto"; records every host copy and every return copy."""
    monkeypatch.setattr(dispatch, "_backend", "auto")
    log = {"to_host": [], "to_device": []}

    def to_host(t):
        log["to_host"].append((t.device, t.dtype))
        return t.to("cpu")

    def to_device(t, device):
        log["to_device"].append((torch.device(device), t.dtype))
        return t.to(device)

    monkeypatch.setattr(dispatch, "_fp64_to_host", to_host)
    monkeypatch.setattr(dispatch, "_fp64_to_device", to_device)
    return log


def _host_warnings(rec):
    return [
        w
        for w in rec
        if issubclass(w.category, UserWarning)
        and "float64 on the neuron device is computed on the host" in str(w.message)
    ]


@pytest.mark.parametrize("op", sorted(FP64_HOST_CALLS))
def test_fp64_host_route_auto(host_spy, op):
    """ "auto", float64 / complex128 on the neuron device: the operands go to the
    host, no kernel runs, the result comes back to the operands' device in
    float64 / complex128, and it matches a float64 CPU reference to float64
    tolerance."""
    call, ref = FP64_HOST_CALLS[op]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = call()
    assert CALLS == [], CALLS
    assert host_spy["to_host"], "no operand was copied to the host"
    assert all(dt in dispatch._FP64_DTYPES for _, dt in host_spy["to_host"])
    assert host_spy["to_device"], "the result did not go back to the device"
    assert all(d == torch.device("cpu") for d, _ in host_spy["to_device"])
    parts = (got.real, got.imag) if isinstance(got, ComplexTensor) else (got,)
    for t in parts:
        assert t.dtype == torch.float64, t.dtype
        assert t.device == torch.device("cpu")
    err = _relerr(got, ref())
    assert err < 1e-12, (op, err)
    ws = _host_warnings(rec)
    assert len(ws) == 1 and f"trnfft {op}:" in str(ws[0].message), [str(w.message) for w in ws]
    assert "pass float32 to stay on device" in str(ws[0].message)


def test_fp64_host_warns_once_per_op(host_spy):
    """One warning per op over repeated calls; another op gets its own."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(3):
            for op in ("fft", "rfft", "complex_gemm", "ComplexLinear"):
                FP64_HOST_CALLS[op][0]()
    ops = sorted(str(w.message).split(":")[0] for w in _host_warnings(rec))
    assert ops == [
        "trnfft ComplexLinear",
        "trnfft complex_gemm",
        "trnfft fft",
        "trnfft rfft",
    ]


def test_fp64_host_route_keeps_grad(host_spy):
    """The host copies are differentiable: gradients reach the device operand."""
    x = _r((2, 64)).requires_grad_(True)
    y = trnfft.rfft(x)
    (y.real.sum() + y.imag.sum()).backward()
    xr = x.detach().clone().requires_grad_(True)
    w = torch.fft.rfft(xr)
    (w.real.sum() + w.imag.sum()).backward()
    assert x.grad is not None and x.grad.dtype == torch.float64
    assert torch.allclose(x.grad, xr.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "op", ["fft", "rfft", "complex_gemm", "complex_mask_apply", "ComplexLinear"]
)
def test_fp64_on_device_override_keeps_device_path(host_spy, monkeypatch, op):
    """TRNFFT_FP64_ON_DEVICE=1: no host copy, no host-route warning; the call
    takes the device torch route as before (no kernel, float64 result)."""
    monkeypatch.setenv("TRNFFT_FP64_ON_DEVICE", "1")
    call, _ = FP64_HOST_CALLS[op]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = call()
    assert host_spy == {"to_host": [], "to_device": []}
    assert not _host_warnings(rec)
    assert CALLS == []
    assert got.real.dtype == torch.float64


@pytest.mark.parametrize("bk", ["nki", "pytorch"])
def test_fp64_host_route_only_under_auto(host_spy, monkeypatch, bk):
    """Backends "nki" (fp32 kernel cast) and "pytorch" (torch where the data
    lives) are unchanged: no host copy, no host-route warning."""
    monkeypatch.setattr(dispatch, "_backend", bk)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for op in ("fft", "complex_gemm", "complex_linear", "ComplexLinear"):
            FP64_HOST_CALLS[op][0]()
    assert host_spy == {"to_host": [], "to_device": []}
    assert not _host_warnings(rec)
    if bk == "nki":
        assert CALLS and all(d <= {torch.float32} for _, d in CALLS)


def test_fp64_host_resident_unchanged(host_spy, monkeypatch):
    """Host-resident float64 (not on the neuron device) keeps host torch: no
    copy, no warning, float64-accurate."""
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "neuron")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = trnfft.fft(_z((3, 64)))
        dispatch.complex_gemm(_x64((4, 6)), _x64((6, 5), 2))
    assert host_spy == {"to_host": [], "to_device": []}
    assert not _host_warnings(rec)
    assert CALLS == []
    assert _relerr(got, torch.fft.fft(_z((3, 64)))) < 1e-12


def test_fp64_host_route_skips_float32(host_spy):
    """float32 on the device stays on the device route (the kernel)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        trnfft.fft(_ct((3, 64), torch.float32))
    assert host_spy == {"to_host": [], "to_device": []}
    assert not _host_warnings(rec)
    assert CALLS


def test_fp64_host_route_require_nki_raises(host_spy, monkeypatch):
    """TRNFFT_REQUIRE_NKI=1 still raises for device-resident float64 under
    "auto", ahead of any host copy."""
    monkeypatch.setenv("TRNFFT_REQUIRE_NKI", "1")
    for op in ("fft", "complex_gemm", "complex_linear", "ComplexLinear"):
        with pytest.raises(NotImplementedError, match="TRNFFT_REQUIRE_NKI"):
            FP64_HOST_CALLS[op][0]()
    assert host_spy == {"to_host": [], "to_device": []}


# ── Result dtype == the pytorch backend's ────────────────────────────────────

MIXED = [
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
    (torch.bfloat16, torch.float64),
    (torch.float16, torch.float64),
    (torch.bfloat16, torch.float32),
    (torch.float16, torch.bfloat16),
]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "op", ["complex_gemm", "complex_linear", "complex_mask_apply", "complex_gemm_bf16"]
)
def test_result_dtype_same_dtype(backend, op, dtype):
    a = _ct((4, 128) if op == "complex_mask_apply" else (4, 6), dtype)
    b = _ct((4, 128) if op == "complex_mask_apply" else (6, 5), dtype, 2)
    if op == "complex_linear":
        b = _ct((3, 6), dtype, 2)
        fn = lambda: dispatch.complex_linear(a, b.real, b.imag)  # noqa: E731
    else:
        fn = lambda: getattr(dispatch, op)(a, b)  # noqa: E731
    got = _same_outcome(fn)
    assert got is not None


@pytest.mark.parametrize("dts", MIXED, ids=lambda d: f"{d[0]}-{d[1]}".replace("torch.", ""))
def test_result_dtype_mixed(backend, dts):
    """Mixed operands: elementwise ops match the pytorch backend (which
    promotes); for the GEMMs, which raise on the pytorch backend for mixed
    operands, a float64 mixture returns torch.promote_types and the other
    mixtures raise as on the pytorch backend."""
    da, db = dts
    m, s = _ct((2, 128), da), _ct((2, 128), db, 2)
    got = _same_outcome(lambda: dispatch.complex_mask_apply(m, s))
    assert got.dtype == torch.promote_types(da, db)
    a, b = _ct((4, 6), da), _ct((6, 5), db, 2)
    w = _ct((3, 6), db, 2)
    for fn in (
        lambda: dispatch.complex_gemm(a, b),
        lambda: dispatch.complex_linear(a, w.real, w.imag),
    ):
        if torch.float64 in dts:
            assert fn().dtype == torch.promote_types(da, db)
        else:
            _same_outcome(fn)


# ── TRNFFT_REQUIRE_NKI ───────────────────────────────────────────────────────


@pytest.fixture
def require_nki(backend, monkeypatch):
    monkeypatch.setenv("TRNFFT_REQUIRE_NKI", "1")


@pytest.mark.parametrize(
    "call",
    [
        lambda: trnfft.fft(_rand((2, 64), torch.float64)),
        lambda: trnfft.fft(_rand((2, 12), torch.float64)),
        lambda: trnfft.ifft(_rand((2, 2048), torch.float64)),
        lambda: trnfft.fft(_rand((2, 64), torch.complex128)),
        lambda: dispatch.complex_gemm(_ct((4, 6), torch.float64), _ct((6, 5), torch.float64)),
        lambda: dispatch.complex_gemm(_ct((4, 6), torch.int32), _ct((6, 5), torch.int32)),
        lambda: dispatch.complex_linear(
            _ct((4, 6), torch.float64), _rand((3, 6), torch.float64), _rand((3, 6), torch.float64)
        ),
        lambda: dispatch.complex_mask_apply(_ct((128,), torch.float64), _ct((128,), torch.float64)),
        lambda: dispatch.complex_gemm_bf16(_ct((4, 6), torch.float64), _ct((6, 5), torch.float64)),
        lambda: ComplexLinear(6, 3).double()(_ct((4, 6), torch.float64)),
    ],
)
def test_require_nki_raises(require_nki, call):
    with pytest.raises(NotImplementedError, match="TRNFFT_REQUIRE_NKI"):
        call()


def test_require_nki_allows_supported_and_double(require_nki, monkeypatch):
    assert trnfft.fft(_rand((2, 64), torch.float32)).dtype == torch.float32
    assert trnfft.fft(_rand((2, 64), torch.bfloat16)).dtype == torch.bfloat16
    # Integer input is promoted by the API (#113) before any kernel.
    assert trnfft.fft(_rand((2, 64), torch.int32)).dtype == torch.float32
    # "double" is the host FP64 path by design, not a fallback.
    monkeypatch.setattr(precision, "_precision", "double")
    assert trnfft.fft(_rand((2, 64), torch.float64)).dtype == torch.float64
    assert trnfft.fft(_rand((2, 2048), torch.float64)).dtype == torch.float64


# ── Coverage: every kernel op was reached ────────────────────────────────────


def test_zz_every_kernel_op_was_exercised():
    """Runs last: each op in `_native.OPS` was reached by some test above, so a
    new kernel with no driving test (or no table entry) is noticed."""
    if not SEEN:
        pytest.skip("run with the rest of this module")
    assert set(SEEN) == set(ALLOWED), sorted(set(ALLOWED) - set(SEEN))
