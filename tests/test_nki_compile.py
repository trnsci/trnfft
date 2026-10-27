"""CPU compile check: every NKI kernel lowers to MLIR that passes verification.

`nki.simulate` does not run the MLIR verifier, so a kernel can pass every
simulator test and still fail to compile for hardware (e.g. integer
`tensor_scalar_arith` operands, mismatched `tensor_tensor` dtypes, or
single-partition access at a non-aligned start partition). This module runs
the NKI frontend (trace -> MLIR -> verify) on CPU for every ``@nki.jit``
kernel in ``trnfft/nki/`` on both trn1 and trn2. It needs ``nki`` but no
Neuron device and no ``neuronx-cc``; it does not run the backend compiler,
so it cannot catch errors raised after MLIR generation.

Shapes mirror what the host wrappers / fft_core drivers pass.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.nki_simulator

from trnfft.nki.dispatch import HAS_NKI  # noqa: E402

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None

try:
    from ml_dtypes import bfloat16 as _np_bf16
except ImportError:
    _np_bf16 = None


def _f32(*shape):
    return np.random.default_rng(0).standard_normal(shape).astype(np.float32)


def _bf16(*shape):
    if _np_bf16 is None:
        pytest.skip("ml_dtypes not available for BF16 example inputs")
    return _f32(*shape).astype(_np_bf16)


def _butterfly_inputs(b, n, stage):
    m = 1 << (stage + 1)
    rows = b * (n // m)
    half = m >> 1
    return (
        (_f32(b, n), _f32(b, n), _f32(rows, half), _f32(rows, half)),
        {"n": n, "stage": stage},
    )


# case id -> (module, kernel name, input factory returning (args, kwargs))
_CASES = {
    "butterfly-s0": ("butterfly", "butterfly_stage_kernel", lambda: _butterfly_inputs(8, 16, 0)),
    "butterfly-last": ("butterfly", "butterfly_stage_kernel", lambda: _butterfly_inputs(8, 16, 3)),
    "butterfly-multitile": (
        "butterfly",
        "butterfly_stage_kernel",
        lambda: _butterfly_inputs(64, 64, 0),
    ),
    "butterfly-kahan-s0": (
        "butterfly",
        "butterfly_stage_kernel_kahan",
        lambda: _butterfly_inputs(8, 16, 0),
    ),
    "butterfly-kahan-last": (
        "butterfly",
        "butterfly_stage_kernel_kahan",
        lambda: _butterfly_inputs(8, 16, 3),
    ),
    "stockham-r4": (
        "stockham",
        "stockham_radix4_stage_kernel",
        lambda: ((_f32(64, 4), _f32(64, 4), _f32(64, 4), _f32(64, 4)), {}),
    ),
    "stockham-r4-multitile": (
        "stockham",
        "stockham_radix4_stage_kernel",
        lambda: ((_f32(256, 4), _f32(256, 4), _f32(256, 4), _f32(256, 4)), {}),
    ),
    "stockham-r8": (
        "stockham",
        "stockham_radix8_w8_kernel",
        lambda: ((_f32(64, 8), _f32(64, 8), _f32(8, 8), _f32(8, 8)), {}),
    ),
    "complex-gemm": (
        "dispatch",
        "_complex_gemm_kernel",
        lambda: ((_f32(128, 128), _f32(128, 128), _f32(128, 128), _f32(128, 128)), {}),
    ),
    "complex-gemm-bf16": (
        "dispatch",
        "_complex_gemm_kernel_bf16",
        lambda: ((_bf16(128, 128), _bf16(128, 128), _bf16(128, 128), _bf16(128, 128)), {}),
    ),
    "complex-linear": (
        "dispatch",
        "_complex_linear_kernel",
        lambda: ((_f32(128, 128), _f32(128, 128), _f32(128, 128), _f32(128, 128)), {}),
    ),
    "complex-mul": (
        "dispatch",
        "_complex_mul_kernel",
        lambda: ((_f32(128, 4), _f32(128, 4), _f32(128, 4), _f32(128, 4)), {}),
    ),
}


@pytest.fixture(autouse=True)
def _frontend_available():
    if not HAS_NKI:
        pytest.skip("nki not importable on this host")
    if source_to_mlir is None:
        pytest.skip("nki.compiler.frontend.source_to_mlir not available in this nki")


@pytest.mark.parametrize("target", ["trn1", "trn2"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_kernel_mlir_verifies(case, target):
    import importlib

    module_name, kernel_name, make_inputs = _CASES[case]
    module = importlib.import_module(f"trnfft.nki.{module_name}")
    kernel = getattr(module, kernel_name)
    assert kernel is not None, f"{module_name}.{kernel_name} is not defined"
    args, kwargs = make_inputs()
    result = source_to_mlir(kernel, *args, **kwargs, target=target, frontend=FrontendKind.TRACER)
    result.verify()
