"""
Dispatch-path selection (native TorchNeuron vs XLA vs CPU), launch routing,
lazy registration of the `trnfft::` custom ops, and the native-path autograd
(`torch.library.register_autograd`).

CPU-only: `torch_neuronx` and `torch_xla` are replaced with stand-ins via
`sys.modules` / `find_spec` monkeypatching, so these run in the default
`pytest -m "not neuron"` job on any host. For the end-to-end native tests the
stand-in `nki_op` registers real `torch.library` custom ops, "cpu" stands in
for the "neuron" device, and the kernels are torch reference functions.
"""

import importlib.util
import subprocess
import sys
import textwrap
import types
import warnings

import numpy as np
import pytest
import torch

import trnfft
import trnfft.nki
from trnfft import ComplexTensor
from trnfft.nki import dispatch

_REAL_FIND_SPEC = importlib.util.find_spec

# Ops registered with nki_op (one per @nki.jit kernel).
EXPECTED_NKI_OPS = {
    "complex_gemm",
    "complex_gemm_bf16",
    "complex_gemm_tn",
    "complex_gemm_t2d",
    "complex_linear",
    "complex_mul",
    "butterfly_stage",
    "butterfly_stage_kahan",
    "stockham_radix4_stage",
    "stockham_radix8_w8",
    "stockham_radix8_w8_t2d",
    "four_step_fft",
}
# Plus the composite FFT op (torch.library.custom_op) that carries the adjoint.
EXPECTED_OPS = EXPECTED_NKI_OPS | {"fft"}

# Source of the stand-in TorchNeuron `torch_neuronx`, shared with the
# fresh-interpreter tests. `nki_op` records each registration and, unless
# `real_ops` is False, registers a real torch.library custom op the way the
# real one does (custom_op + register_fake), so register_autograd works.
_FAKE_NEURONX_SRC = textwrap.dedent(
    """
    import importlib.machinery
    import types

    import torch

    def make_fake_torch_neuronx(recorded=None, device_count=1, real_ops=True):
        mod = types.ModuleType("torch_neuronx")
        mod.__spec__ = importlib.machinery.ModuleSpec("torch_neuronx", None)

        def nki_op(name, fn=None, mutates_args={}, fake_fn=None):
            def dec(f):
                if recorded is not None:
                    recorded[name] = (f, mutates_args, fake_fn)
                if not real_ops:
                    return f
                op = torch.library.custom_op(name, f, mutates_args=mutates_args)
                op.register_fake(fake_fn if fake_fn is not None else f)
                return op

            return dec if fn is None else dec(fn)

        mod.nki_op = nki_op
        mod.device_count = lambda: device_count
        return mod
    """
)
_fakes = {}
exec(_FAKE_NEURONX_SRC, _fakes)
_fake_native_torch_neuronx = _fakes["make_fake_torch_neuronx"]


def _fake_xla_torch_neuronx():
    """Stand-in for the XLA-lineage `torch_neuronx` (no `nki_op`)."""
    mod = types.ModuleType("torch_neuronx")
    mod.trace = lambda *a, **k: None
    return mod


def _set_torch_xla_present(monkeypatch, present: bool):
    def find_spec(name, *args, **kwargs):
        if name == "torch_xla":
            return object() if present else None
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)


def _accept_neuron_device(monkeypatch):
    # Off-hardware, "neuron" is not a registered torch device type.
    real_device = torch.device

    def device(spec, *args, **kwargs):
        if spec == dispatch._NEURON_DEVICE:
            return real_device("cpu")
        return real_device(spec, *args, **kwargs)

    monkeypatch.setattr(dispatch.torch, "device", device)


@pytest.fixture(autouse=True)
def reset_dispatch_state(monkeypatch):
    """Each test sees an undetected path and an unloaded op table."""
    monkeypatch.setattr(dispatch, "_dispatch_path", None)
    monkeypatch.setattr(dispatch, "_native_ops", None)
    monkeypatch.setattr(dispatch, "_xla_deprecation_warned", False)
    monkeypatch.delenv("TRNSCI_NKI_DISPATCH", raising=False)
    monkeypatch.delenv("TRNSCI_XLA_DEPRECATION", raising=False)
    yield


@pytest.fixture(scope="module")
def native_module():
    """Import `trnfft.nki._native` once (registering real custom ops) under a
    stand-in torch_neuronx. Registration is process-global, so this happens
    once per module; the fresh-interpreter tests cover first-use loading."""
    recorded = {}
    saved = sys.modules.get("torch_neuronx")
    sys.modules["torch_neuronx"] = _fake_native_torch_neuronx(recorded=recorded)
    try:
        sys.modules.pop("trnfft.nki._native", None)
        import trnfft.nki._native as native
    finally:
        if saved is None:
            sys.modules.pop("torch_neuronx", None)
        else:
            sys.modules["torch_neuronx"] = saved
    return native, recorded


# ── Path selection ───────────────────────────────────────────────────────────


class TestDispatchPathSelection:
    def test_native_when_nki_op_and_device(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "native"

    def test_native_probe_does_not_touch_torch_xla(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        # Any import of torch_xla would raise ImportError; any spec lookup
        # would fail the test.
        monkeypatch.setitem(sys.modules, "torch_xla", None)

        def find_spec(name, *args, **kwargs):
            assert name != "torch_xla", "native probe must not look up torch_xla"
            return _REAL_FIND_SPEC(name, *args, **kwargs)

        monkeypatch.setattr(dispatch.importlib.util, "find_spec", find_spec)
        assert dispatch._get_dispatch_path() == "native"

    def test_xla_when_torch_neuronx_has_no_nki_op(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_xla_when_native_has_no_device(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "torch_neuronx", _fake_native_torch_neuronx(device_count=0)
        )
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        assert dispatch._get_dispatch_path() == "xla"

    def test_not_native_when_neuron_device_type_unregistered(self, monkeypatch):
        # torch_neuronx imported but "neuron" is not a torch device type
        # (e.g. another backend owns PrivateUse1).
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_not_native_when_device_count_raises(self, monkeypatch):
        mod = _fake_native_torch_neuronx()

        def boom():
            raise RuntimeError("no runtime")

        mod.device_count = boom
        monkeypatch.setitem(sys.modules, "torch_neuronx", mod)
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_cpu_when_nothing_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", None)  # import raises
        _set_torch_xla_present(monkeypatch, False)
        assert dispatch._get_dispatch_path() == "cpu"

    def test_stand_in_torch_xla_without_spec_counts_as_installed(self, monkeypatch):
        # find_spec raises ValueError for a sys.modules entry with no __spec__
        # (the stand-ins the device-return tests install).
        monkeypatch.setitem(sys.modules, "torch_neuronx", None)
        monkeypatch.setitem(sys.modules, "torch_xla", types.ModuleType("torch_xla"))
        assert dispatch._get_dispatch_path() == "xla"

    def test_detection_is_cached(self, monkeypatch):
        calls = []

        def detect():
            calls.append(1)
            return "xla"

        monkeypatch.setattr(dispatch, "_detect_dispatch_path", detect)
        assert dispatch._get_dispatch_path() == "xla"
        assert dispatch._get_dispatch_path() == "xla"
        assert len(calls) == 1

    def test_import_does_not_detect(self):
        # Detection is lazy: a fresh interpreter importing trnfft has no
        # cached path until the first launch.
        code = (
            "import trnfft; from trnfft.nki import dispatch; assert dispatch._dispatch_path is None"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class TestDispatchOverride:
    """TRNSCI_NKI_DISPATCH=auto|native|xla|cpu (suite-wide) pins the path."""

    def _both_lineages(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)

    def test_auto_prefers_native(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "auto")
        assert dispatch._get_dispatch_path() == "native"

    def test_force_xla_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "XLA")
        assert dispatch._get_dispatch_path() == "xla"

    def test_force_cpu_when_both_available(self, monkeypatch):
        self._both_lineages(monkeypatch)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cpu")
        assert dispatch._get_dispatch_path() == "cpu"

    def test_force_native_without_backend_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "native")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=native"):
            dispatch._get_dispatch_path()

    def test_force_xla_without_torch_xla_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, False)
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "xla")
        with pytest.raises(RuntimeError, match="TRNSCI_NKI_DISPATCH=xla"):
            dispatch._get_dispatch_path()

    def test_unknown_value_raises(self, monkeypatch):
        monkeypatch.setenv("TRNSCI_NKI_DISPATCH", "cuda")
        with pytest.raises(ValueError, match="auto, native, xla, or cpu"):
            dispatch._get_dispatch_path()


class TestXlaDeprecationHook:
    """Dormant DeprecationWarning for the XLA path (trnsci#35)."""

    def _xla_only(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_xla_torch_neuronx())
        _set_torch_xla_present(monkeypatch, True)

    def test_off_by_default(self, monkeypatch):
        self._xla_only(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert dispatch._get_dispatch_path() == "xla"

    def test_fires_once_when_enabled(self, monkeypatch):
        self._xla_only(monkeypatch)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert dispatch._get_dispatch_path() == "xla"
            assert dispatch._get_dispatch_path() == "xla"
            # Even a re-detection does not warn a second time.
            monkeypatch.setattr(dispatch, "_dispatch_path", None)
            assert dispatch._get_dispatch_path() == "xla"
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert str(dep[0].message) == (
            "the PyTorch/XLA path is deprecated on Trn2/Trn3; use TorchNeuron (native). "
            "Trn1/Inf2 remain supported in maintenance mode."
        )

    def test_not_fired_on_native_path(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_neuronx", _fake_native_torch_neuronx())
        _accept_neuron_device(monkeypatch)
        _set_torch_xla_present(monkeypatch, True)
        monkeypatch.setenv("TRNSCI_XLA_DEPRECATION", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert dispatch._get_dispatch_path() == "native"


# ── Launch routing ───────────────────────────────────────────────────────────


class TestLaunchRouting:
    def test_native_launch_uses_custom_op_not_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        seen = {}

        def op(a, b, n, stage):
            seen["args"] = (a, b, n, stage)
            return a + b, a - b

        monkeypatch.setattr(dispatch, "_native_ops", {"fake": op})

        def no_xla(*t):
            raise AssertionError("_to_xla must not run on the native path")

        def no_kernel(*t):
            raise AssertionError("raw kernel must not be called on the native path")

        monkeypatch.setattr(dispatch, "_to_xla", no_xla)
        out = dispatch._launch(no_kernel, "fake", torch.ones(2, 1), torch.ones(2, 1), 16, 3)
        assert isinstance(out, tuple) and len(out) == 2
        assert torch.equal(out[0], torch.full((2, 1), 2.0))
        assert out[0].device.type == "cpu"
        # Python ints (butterfly n / stage) pass through untouched.
        assert seen["args"][2:] == (16, 3)

    def test_xla_launch_calls_kernel_via_to_xla(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        moved = []

        def fake_to_xla(*tensors):
            moved.append(len(tensors))
            return list(tensors), tensors[0].device

        monkeypatch.setattr(dispatch, "_to_xla", fake_to_xla)

        def no_ops():
            raise AssertionError("native ops must not load on the XLA path")

        monkeypatch.setattr(dispatch, "_get_native_ops", no_ops)
        out = dispatch._launch(
            lambda x, y, n: (x * n, y), "unused", torch.ones(4, 1), torch.ones(4, 1), 3
        )
        assert moved == [2]
        assert torch.equal(out[0], torch.full((4, 1), 3.0))

    def test_cpu_launch_keeps_pre_native_behaviour(self, monkeypatch):
        # No device path: the launch goes through `_to_xla`, which raises
        # when torch_xla is absent — the same failure as before.
        monkeypatch.setattr(dispatch, "_dispatch_path", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)
        with pytest.raises(ImportError):
            dispatch._launch(lambda x: x, "unused", torch.ones(1, 1))

    def test_kernel_device(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", None)  # must not be imported
        assert dispatch._kernel_device() == torch.device("cpu")

        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        fake_xla = types.ModuleType("torch_xla")
        fake_xla.device = lambda: torch.device("meta")
        monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
        assert dispatch._kernel_device() == torch.device("meta")


# ── Lazy registration ────────────────────────────────────────────────────────


def _run_fresh(body: str):
    code = _FAKE_NEURONX_SRC + "\n" + textwrap.dedent(body)
    subprocess.run([sys.executable, "-c", code], check=True)


class TestLazyRegistration:
    def test_import_does_not_register_ops(self):
        # Fresh interpreter: importing trnfft (and every NKI module) must not
        # load `_native` or import torch_neuronx / torch_xla.
        code = (
            "import sys, trnfft, trnfft.nki.dispatch, trnfft.nki.autograd, "
            "trnfft.nki.multicore, trnfft.fft_core, trnfft.nn; "
            "bad = [m for m in ('trnfft.nki._native', 'torch_neuronx', 'torch_xla') "
            "if m in sys.modules]; "
            "assert not bad, bad"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_ops_registered_on_first_native_use(self):
        _run_fresh(
            f"""
            import sys
            recorded = {{}}
            sys.modules["torch_neuronx"] = make_fake_torch_neuronx(recorded=recorded)
            from trnfft.nki import dispatch
            assert "trnfft.nki._native" not in sys.modules
            assert recorded == {{}}

            ops = dispatch._get_native_ops()

            assert set(ops) == set({sorted(EXPECTED_OPS)!r})
            assert set(recorded) == {{f"trnfft::{{n}}" for n in {sorted(EXPECTED_NKI_OPS)!r}}}
            assert all(m == {{}} for _, m, _ in recorded.values())
            assert all(f is not None for _, _, f in recorded.values())
            for name in {sorted(EXPECTED_OPS)!r}:
                assert hasattr(torch.ops.trnfft, name), name

            # Second call reuses the table; no re-registration.
            recorded.clear()
            assert dispatch._get_native_ops() is ops
            assert recorded == {{}}
            assert "torch_xla" not in sys.modules
            """
        )

    def test_op_signatures_build_a_schema(self, native_module):
        # nki_op derives the schema with torch.library.infer_schema; check
        # every wrapper is fully annotated (Tensor args, (Tensor, Tensor) out).
        _, recorded = native_module
        assert set(recorded) == {f"trnfft::{n}" for n in EXPECTED_NKI_OPS}
        for name, (fn, mutates, _) in recorded.items():
            schema = torch.library.infer_schema(fn, mutates_args=mutates)
            assert schema.endswith("-> (Tensor, Tensor)"), (name, schema)
            n_in = 7 if name == "trnfft::four_step_fft" else 4
            assert schema.split("->")[0].count("Tensor") == n_in, (name, schema)
        bfly = torch.library.infer_schema(recorded["trnfft::butterfly_stage"][0], mutates_args={})
        assert "SymInt n, SymInt stage" in bfly
        fft_schema = str(torch.ops.trnfft.fft.default._schema)
        assert "bool inverse, str precision" in fft_schema

    def test_fake_fns_give_kernel_output_shapes(self, native_module):
        _, recorded = native_module

        def fake(name):
            return recorded[f"trnfft::{name}"][2]

        def meta(*shape, dtype=torch.float32):
            return torch.empty(*shape, dtype=dtype, device="meta")

        a, b = meta(128, 64), meta(64, 256)
        cr, ci = fake("complex_gemm")(a, a, b, b)
        assert cr.shape == ci.shape == (128, 256) and cr.dtype == torch.float32
        a16, b16 = meta(128, 64, dtype=torch.bfloat16), meta(64, 256, dtype=torch.bfloat16)
        cr, _ = fake("complex_gemm_bf16")(a16, a16, b16, b16)
        assert cr.shape == (128, 256) and cr.dtype == torch.float32  # FP32 PSUM output
        at, w = meta(3 * 64, 128), meta(64, 64)  # 3 stacked (K=64, M=128) blocks
        cr, ci = fake("complex_gemm_tn")(at, at, w, w)
        assert cr.shape == ci.shape == (3 * 128, 64) and cr.dtype == torch.float32
        x, w = meta(128, 64), meta(32, 64)
        assert fake("complex_linear")(x, x, w, w)[0].shape == (128, 32)
        assert fake("complex_mul")(a, a, a, a)[1].shape == (128, 64)
        assert fake("butterfly_stage")(meta(8, 16), meta(8, 16), meta(64, 1), meta(64, 1), 16, 0)[
            0
        ].shape == (8, 16)
        assert fake("butterfly_stage_kahan")(
            meta(8, 16), meta(8, 16), meta(4, 8), meta(4, 8), 16, 3
        )[1].shape == (8, 16)
        assert fake("stockham_radix4_stage")(*[meta(256, 4)] * 4)[0].shape == (256, 4)
        assert fake("stockham_radix8_w8")(meta(64, 8), meta(64, 8), meta(8, 8), meta(8, 8))[
            0
        ].shape == (64, 8)
        fs = fake("four_step_fft")(
            meta(9, 4096), meta(9, 4096), meta(128, 128), *[meta(64, 512)] * 2, *[meta(64, 64)] * 2
        )
        assert fs[0].shape == fs[1].shape == (9, 4096)
        yr, yi = torch.ops.trnfft.fft(meta(3, 256), meta(3, 256), False, "fast")
        assert yr.shape == yi.shape == (3, 256)


# ── Native path end to end on CPU (stand-in device, reference kernels) ──────


def _gemm_ref(a_re, a_im, b_re, b_im):
    return a_re @ b_re - a_im @ b_im, a_re @ b_im + a_im @ b_re


def _linear_ref(x_re, x_im, w_re, w_im):
    return _gemm_ref(x_re, x_im, w_re.t(), w_im.t())


def _mul_ref(a_re, a_im, b_re, b_im):
    return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re


def _four_step_ref(x_re, x_im, w1_blk, t_re, t_im, w2_re, w2_im):
    # four_step_fft_kernel math: X[b] as (N1, N2), Y = (X^T W1) * T, O = W2^T Y.
    b, n = x_re.shape
    n1 = w1_blk.shape[0] // 2
    n2 = n // n1
    W1 = torch.complex(w1_blk[:n1, :n1], w1_blk[:n1, n1:])
    X = torch.complex(x_re, x_im).reshape(b, n1, n2)
    Y = torch.einsum("bij,ik->bjk", X, W1) * torch.complex(t_re[:n2, :n1], t_im[:n2, :n1])
    W2 = torch.complex(w2_re[:n2, :n2], w2_im[:n2, :n2])
    O = torch.einsum("jl,bjk->blk", W2, Y).reshape(b, n)
    return O.real.contiguous(), O.imag.contiguous()


@pytest.fixture
def native_cpu(monkeypatch, native_module):
    """Native path selected, "cpu" standing in for "neuron", torch_xla unusable,
    reference kernels in place of the @nki.jit ones."""
    from trnfft import fft_core as fc
    from trnfft.nki import autograd as ag

    native, _ = native_module
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_native_ops", native.OPS)
    monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "torch_xla", None)  # any import raises
    monkeypatch.setattr(dispatch, "HAS_NKI", True)
    monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.setattr(fc, "_use_nki", lambda: True)
    # Other suites can leave a global precision behind; these tests need "fast".
    from trnfft import precision

    monkeypatch.setattr(precision, "_precision", "fast")
    # Off-NKI the butterfly kernels are undefined; the GEMM paths under test
    # import but never call them.
    from trnfft.nki import butterfly

    monkeypatch.setattr(butterfly, "butterfly_stage_kernel", None, raising=False)
    monkeypatch.setattr(butterfly, "butterfly_stage_kernel_kahan", None, raising=False)

    calls = {"gemm": 0, "linear": 0, "mul": 0, "four_step": 0}

    def counted(name, fn):
        def k(*args):
            calls[name] += 1
            return fn(*args)

        return k

    monkeypatch.setattr(dispatch, "_complex_gemm_kernel", counted("gemm", _gemm_ref), raising=False)
    monkeypatch.setattr(
        dispatch, "_complex_linear_kernel", counted("linear", _linear_ref), raising=False
    )
    monkeypatch.setattr(dispatch, "_complex_mul_kernel", counted("mul", _mul_ref), raising=False)
    from trnfft.nki import four_step

    monkeypatch.setattr(
        four_step, "four_step_fft_kernel", counted("four_step", _four_step_ref), raising=False
    )

    # The XLA-path autograd.Functions must not run on the native path.
    def no_function(*a, **k):
        raise AssertionError("torch.autograd.Function used on the native path")

    for cls in (ag._ComplexGEMMFn, ag._ComplexLinearFn, ag._ComplexMulFn, ag._FFTFn):
        monkeypatch.setattr(cls, "apply", no_function)
    return calls


def _rand(*shape, seed=0, grad=True):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float64).float()
    return t.requires_grad_(grad)


class TestNativeAutogradCPU:
    """register_autograd on the native ops reproduces the analytic adjoints."""

    @staticmethod
    def _check_grads(fn_trnfft, fn_ref, inputs):
        out_re, out_im = fn_trnfft(*inputs)
        g = torch.Generator().manual_seed(7)
        w_re, w_im = torch.randn(out_re.shape, generator=g), torch.randn(out_im.shape, generator=g)
        (out_re * w_re + out_im * w_im).sum().backward()
        got = [t.grad.clone() for t in inputs]

        ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
        r_re, r_im = fn_ref(*ref_inputs)
        torch.testing.assert_close(out_re.detach(), r_re.detach(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out_im.detach(), r_im.detach(), rtol=1e-5, atol=1e-5)
        (r_re * w_re + r_im * w_im).sum().backward()
        for a, b in zip(got, (t.grad for t in ref_inputs), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_complex_gemm(self, native_cpu):
        from trnfft.nki.autograd import complex_gemm_autograd

        inputs = [_rand(8, 16, seed=i) if i < 2 else _rand(16, 4, seed=i) for i in range(4)]
        self._check_grads(complex_gemm_autograd, _gemm_ref, inputs)
        assert native_cpu["gemm"] == 1

    def test_complex_linear(self, native_cpu):
        from trnfft.nki.autograd import complex_linear_autograd

        inputs = [_rand(8, 16, seed=i) if i < 2 else _rand(4, 16, seed=i) for i in range(4)]
        self._check_grads(complex_linear_autograd, _linear_ref, inputs)
        assert native_cpu["linear"] == 1

    def test_complex_mul(self, native_cpu):
        from trnfft.nki.autograd import complex_mul_autograd

        inputs = [_rand(128, 4, seed=i) for i in range(4)]
        self._check_grads(complex_mul_autograd, _mul_ref, inputs)
        assert native_cpu["mul"] == 1

    def test_complex_linear_layer_trains(self, native_cpu):
        from trnfft.nn import ComplexLinear

        layer = ComplexLinear(16, 8)
        x = ComplexTensor(_rand(4, 16, seed=1, grad=False), _rand(4, 16, seed=2, grad=False))
        y = layer(x)
        (y.real.square().sum() + y.imag.square().sum()).backward()
        assert native_cpu["linear"] >= 1
        assert all(p.grad is not None for p in layer.parameters())

    @pytest.mark.parametrize("n", [64, 2048])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_fft_op_forward_and_backward(self, native_cpu, monkeypatch, n, inverse):
        # n=64: single DFT-GEMM; n=2048: batch-fused four-step (one launch).
        from trnfft import fft_core as fc

        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1024)
        monkeypatch.setattr(fc, "_FOUR_STEP_MAX", 4096)
        x_re, x_im = _rand(4, n, seed=1), _rand(4, n, seed=2)
        fn = trnfft.ifft if inverse else trnfft.fft
        y = fn(ComplexTensor(x_re, x_im))
        assert native_cpu["four_step" if n > 1024 else "gemm"] >= 1

        xc = torch.complex(x_re.detach().double(), x_im.detach().double()).requires_grad_(True)
        ref = (torch.fft.ifft if inverse else torch.fft.fft)(xc)
        torch.testing.assert_close(
            y.real.detach().double(), ref.real.detach(), rtol=1e-4, atol=1e-4
        )

        g = torch.Generator().manual_seed(3)
        w_re, w_im = torch.randn(4, n, generator=g), torch.randn(4, n, generator=g)
        (y.real * w_re + y.imag * w_im).sum().backward()
        (ref.real * w_re.double() + ref.imag * w_im.double()).sum().backward()
        torch.testing.assert_close(x_re.grad.double(), xc.grad.real, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(x_im.grad.double(), xc.grad.imag, rtol=1e-4, atol=1e-4)


class TestNativeStockhamCPU:
    """The row-chunked Stockham stage launches go through the native op on a
    device-resident buffer, with no torch_xla (trnfft#83 helper kept)."""

    def test_radix4_chunked_matches_numpy(self, native_cpu, monkeypatch, native_module):
        from trnfft import fft_core as fc
        from trnfft.nki import stockham

        launches = []

        def fake_r4(x_re, x_im, tw_re, tw_im):
            launches.append(x_re.shape[0])
            a_re = x_re * tw_re - x_im * tw_im
            a_im = x_re * tw_im + x_im * tw_re
            k = torch.arange(4, dtype=torch.float64)
            ang = -2.0 * np.pi * torch.outer(k, k) / 4
            w_re, w_im = torch.cos(ang).float(), torch.sin(ang).float()
            return a_re @ w_re - a_im @ w_im, a_re @ w_im + a_im @ w_re

        monkeypatch.setattr(stockham, "stockham_radix4_stage_kernel", fake_r4, raising=False)
        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", "128")
        n, batch = 256, 4
        x_re, x_im = _rand(batch, n, seed=4, grad=False), _rand(batch, n, seed=5, grad=False)
        y = fc._fft_via_stockham_nki(ComplexTensor(x_re, x_im), inverse=False)
        ref = np.fft.fft(x_re.double().numpy() + 1j * x_im.double().numpy())
        np.testing.assert_allclose(y.real.numpy(), ref.real, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(y.imag.numpy(), ref.imag, rtol=1e-4, atol=1e-3)
        # total_groups = 4 * 256 / 4 = 256 rows per stage, cap 128 -> 2 launches x 4 stages.
        assert launches == [128] * 8
