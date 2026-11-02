"""No NKI kernel is reached with an operand dtype it does not take (trnsci#35).

float64 (and integer, bool, complex) tensors reached the FP32 kernels as is
and came back inf/NaN or garbage on Trn2 (#108, #131; the same class in the
other trnsci libraries). ``dispatch.KERNEL_DTYPES`` lists what each kernel
takes; ``_launch`` / ``_simulate_kernel`` / every ``nki_op`` body enforce it,
and the public wrappers cast float64 to float32 (result cast back) or run
other dtypes in torch first.

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

import pytest
import torch

import trnfft
import trnfft.fft_core as fc
import trnfft.nki as trnfft_nki
import trnfft.nki.dispatch as dispatch
from trnfft import ComplexTensor, precision
from trnfft.nn import ComplexLinear

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


class _StubOp:
    """Stand-in for a registered `nki_op`: dtype check + zero outputs."""

    def __init__(self, name, fn, fake_fn):
        self.name = name.split("::", 1)[1]
        self.fn = fn
        self.fake_fn = fake_fn

    def __call__(self, *args):
        dts = {a.dtype for a in args if isinstance(a, torch.Tensor)}
        SEEN.setdefault(self.name, set()).update(dts)
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
    VIOLATIONS.clear()
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
    if dtype in (torch.bfloat16, torch.float16) and prec in _LOWP and n <= fc._LOWP_GEMM_MAX:
        # The BF16-GEMM modes return their FP32 PSUM; the pytorch backend
        # returns the input dtype (needs-decision, trnfft#137).
        y = trnfft.fft(x)
        assert y.dtype == torch.float32
        pytest.xfail("half input in a BF16-GEMM mode returns float32 (trnfft#137)")
    _same_outcome(trnfft.fft, x)


_LOWP = ("bf16", "bf16_refined", "ozaki", "ozaki_hq")


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
    """#131 / #132: float64 runs the fp32 kernel and returns float64."""
    layer = ComplexLinear(6, 3).double()
    x = ComplexTensor(_rand((4, 6), torch.float64, 1), _rand((4, 6), torch.float64, 2))
    assert layer(x).real.dtype == torch.float64
    assert SEEN["complex_linear"] - _FP32_HALF == set()


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


# ── TRNFFT_REQUIRE_NKI ───────────────────────────────────────────────────────


@pytest.fixture
def require_nki(backend, monkeypatch):
    monkeypatch.setenv("TRNFFT_REQUIRE_NKI", "1")


@pytest.mark.parametrize(
    "call",
    [
        lambda: trnfft.fft(_rand((2, 64), torch.float64)),
        lambda: trnfft.fft(_rand((2, 12), torch.float64)),
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
