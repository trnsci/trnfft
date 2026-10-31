"""
Native TorchNeuron registration of the trnfft NKI kernels.

On the TorchNeuron lineage (Neuron SDK 2.33+, torch-neuronx 2.14,
PyTorch 2.14, Trn2/Trn3), Trainium is a native PyTorch `neuron` device
and an `@nki.jit` kernel is exposed to PyTorch as a custom operator via
`torch_neuronx.nki_op`. Each wrapper below calls an existing kernel
unchanged; the op schema is inferred from the annotations. Complex values
stay split into real/imag tensors (no complex dtype on the device), so
every op takes and returns real/imag pairs.

Autograd: the complex GEMM / linear / elementwise-multiply ops and the
composite ``trnfft::fft`` op get their analytic adjoints through
`torch.library.register_autograd`, which replaces the
`torch.autograd.Function` wrappers of `autograd.py` on this path. The
backward formulas are shared with those wrappers (the XLA path keeps them).

``trnfft::fft`` is a plain `torch.library.custom_op`, not an `nki_op`: it is
a host-side composite (several kernel launches plus host reshapes) whose
only purpose is to carry the FFT-level adjoint (IFFT * n / FFT / n), the
same contract as `autograd._FFTFn`.

Importing this module registers the ops, so `dispatch.py` imports it only
after selecting the native backend. CPU-only installs and the Trn1 / XLA
lineage (whose `torch_neuronx` has no `nki_op`) never load it.

This module deliberately does not use `from __future__ import annotations`:
`nki_op` builds the op schema from the live annotation objects.

The kernels are looked up on their modules at call time rather than
imported by name, so registration itself does not trace or compile
anything; the first call per input shape does.
"""

import torch
from torch_neuronx import nki_op

from . import autograd as _ag
from . import butterfly as _bfly
from . import dispatch as _d
from . import four_step as _fs
from . import stockham as _st

Pair = tuple[torch.Tensor, torch.Tensor]


# Fake (meta) implementations: output shape/dtype only, no kernel trace.
# Used by torch.compile tracing and for meta-device inputs.


def _fake_pair(shape, dtype: torch.dtype, ref: torch.Tensor) -> Pair:
    return ref.new_empty(shape, dtype=dtype), ref.new_empty(shape, dtype=dtype)


def _fake_like(x: torch.Tensor) -> Pair:
    return torch.empty_like(x), torch.empty_like(x)


# Shared autograd plumbing for the four-tensor complex kernels.


def _save_inputs(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _register_backward(op, backward_math):
    def backward(ctx, grad_re, grad_im):
        return backward_math(*ctx.saved_tensors, grad_re, grad_im)

    op.register_autograd(backward, setup_context=_save_inputs)


# ── Complex GEMM / linear / elementwise multiply ─────────────────────────────


@nki_op(
    "trnfft::complex_gemm",
    mutates_args={},
    fake_fn=lambda a_real, a_imag, b_real, b_imag: _fake_pair(
        (a_real.shape[0], b_real.shape[1]), a_real.dtype, a_real
    ),
)
def complex_gemm(
    a_real: torch.Tensor, a_imag: torch.Tensor, b_real: torch.Tensor, b_imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _d._complex_gemm_kernel(a_real, a_imag, b_real, b_imag)


@nki_op(
    "trnfft::complex_gemm_t2d",
    mutates_args={},
    fake_fn=lambda a_real, a_imag, b_real, b_imag: _fake_pair(
        (a_real.shape[0], b_real.shape[1]), a_real.dtype, a_real
    ),
)
def complex_gemm_t2d(
    a_real: torch.Tensor, a_imag: torch.Tensor, b_real: torch.Tensor, b_imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TRNFFT_PE_TRANSPOSE=0: the load_transpose2d GEMM (trnfft#97).
    return _d._complex_gemm_t2d_kernel(a_real, a_imag, b_real, b_imag)


@nki_op(
    "trnfft::complex_gemm_tn",
    mutates_args={},
    fake_fn=lambda at_real, at_imag, b_real, b_imag: _fake_pair(
        (at_real.shape[0] // b_real.shape[0] * at_real.shape[1], b_real.shape[1]),
        at_real.dtype,
        at_real,
    ),
)
def complex_gemm_tn(
    at_real: torch.Tensor, at_imag: torch.Tensor, b_real: torch.Tensor, b_imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _d._complex_gemm_tn_kernel(at_real, at_imag, b_real, b_imag)


@nki_op(
    "trnfft::complex_gemm_bf16",
    mutates_args={},
    fake_fn=lambda a_real, a_imag, b_real, b_imag: _fake_pair(
        (a_real.shape[0], b_real.shape[1]), torch.float32, a_real
    ),
)
def complex_gemm_bf16(
    a_real: torch.Tensor, a_imag: torch.Tensor, b_real: torch.Tensor, b_imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _d._complex_gemm_kernel_bf16(a_real, a_imag, b_real, b_imag)


@nki_op(
    "trnfft::complex_linear",
    mutates_args={},
    fake_fn=lambda x_real, x_imag, w_real, w_imag: _fake_pair(
        (x_real.shape[0], w_real.shape[0]), x_real.dtype, x_real
    ),
)
def complex_linear(
    x_real: torch.Tensor, x_imag: torch.Tensor, w_real: torch.Tensor, w_imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _d._complex_linear_kernel(x_real, x_imag, w_real, w_imag)


@nki_op(
    "trnfft::complex_mul",
    mutates_args={},
    fake_fn=lambda a_real, a_imag, b_real, b_imag: _fake_like(a_real),
)
def complex_mul(
    a_real: torch.Tensor, a_imag: torch.Tensor, b_real: torch.Tensor, b_imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _d._complex_mul_kernel(a_real, a_imag, b_real, b_imag)


_register_backward(complex_gemm, _ag.complex_gemm_backward)
_register_backward(complex_gemm_t2d, _ag.complex_gemm_backward)
_register_backward(complex_linear, _ag.complex_linear_backward)
_register_backward(complex_mul, _ag.complex_mul_backward)


# ── FFT stage kernels ────────────────────────────────────────────────────────


@nki_op(
    "trnfft::butterfly_stage",
    mutates_args={},
    fake_fn=lambda x_re, x_im, tw_re, tw_im, n, stage: _fake_like(x_re),
)
def butterfly_stage(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    tw_re: torch.Tensor,
    tw_im: torch.Tensor,
    n: int,
    stage: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _bfly.butterfly_stage_kernel(x_re, x_im, tw_re, tw_im, n, stage)


@nki_op(
    "trnfft::butterfly_stage_kahan",
    mutates_args={},
    fake_fn=lambda x_re, x_im, tw_re, tw_im, n, stage: _fake_like(x_re),
)
def butterfly_stage_kahan(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    tw_re: torch.Tensor,
    tw_im: torch.Tensor,
    n: int,
    stage: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _bfly.butterfly_stage_kernel_kahan(x_re, x_im, tw_re, tw_im, n, stage)


@nki_op(
    "trnfft::stockham_radix4_stage",
    mutates_args={},
    fake_fn=lambda x_re, x_im, tw_re, tw_im: _fake_like(x_re),
)
def stockham_radix4_stage(
    x_re: torch.Tensor, x_im: torch.Tensor, tw_re: torch.Tensor, tw_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _st.stockham_radix4_stage_kernel(x_re, x_im, tw_re, tw_im)


@nki_op(
    "trnfft::stockham_radix8_w8",
    mutates_args={},
    fake_fn=lambda a_re, a_im, w8_re, w8_im: _fake_like(a_re),
)
def stockham_radix8_w8(
    a_re: torch.Tensor, a_im: torch.Tensor, w8_re: torch.Tensor, w8_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _st.stockham_radix8_w8_kernel(a_re, a_im, w8_re, w8_im)


@nki_op(
    "trnfft::stockham_radix8_w8_t2d",
    mutates_args={},
    fake_fn=lambda a_re, a_im, w8_re, w8_im: _fake_like(a_re),
)
def stockham_radix8_w8_t2d(
    a_re: torch.Tensor, a_im: torch.Tensor, w8_re: torch.Tensor, w8_im: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TRNFFT_PE_TRANSPOSE=0: the load_transpose2d W_8 kernel (trnfft#97).
    return _st.stockham_radix8_w8_t2d_kernel(a_re, a_im, w8_re, w8_im)


@nki_op(
    "trnfft::four_step_fft",
    mutates_args={},
    fake_fn=lambda x_re, x_im, w1_blk, t_re, t_im, w2_re, w2_im: _fake_like(x_re),
)
def four_step_fft(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    w1_blk: torch.Tensor,
    t_re: torch.Tensor,
    t_im: torch.Tensor,
    w2_re: torch.Tensor,
    w2_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fs.four_step_fft_kernel(x_re, x_im, w1_blk, t_re, t_im, w2_re, w2_im)


# ── Composite FFT (carries the FFT-level adjoint) ────────────────────────────


def _fresh(t: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
    # A custom op's outputs may not alias its inputs.
    if any(t.untyped_storage().data_ptr() == i.untyped_storage().data_ptr() for i in inputs):
        return t.clone()
    return t


@torch.library.custom_op("trnfft::fft", mutates_args=())
def fft(
    x_real: torch.Tensor, x_imag: torch.Tensor, inverse: bool, precision: str
) -> tuple[torch.Tensor, torch.Tensor]:
    from ..complex import ComplexTensor
    from ..fft_core import _cooley_tukey_nki_nograd

    y = _cooley_tukey_nki_nograd(
        ComplexTensor(x_real, x_imag), inverse=inverse, precision=precision
    )
    return _fresh(y.real, x_real, x_imag), _fresh(y.imag, x_real, x_imag)


@fft.register_fake
def _fft_fake(x_real, x_imag, inverse, precision):
    return _fake_like(x_real)


def _fft_setup_context(ctx, inputs, output):
    x_real, _, inverse, precision = inputs
    ctx.inverse = inverse
    ctx.precision = precision
    ctx.n = x_real.shape[-1]


def _fft_backward(ctx, grad_re, grad_im):
    # Same adjoint as autograd._FFTFn: FFT -> IFFT(dy) * n, IFFT -> FFT(dy) / n.
    g_re, g_im = fft(grad_re, grad_im, not ctx.inverse, ctx.precision)
    scale = (1.0 / ctx.n) if ctx.inverse else float(ctx.n)
    return g_re * scale, g_im * scale, None, None


fft.register_autograd(_fft_backward, setup_context=_fft_setup_context)


# Keyed by the short op name used in `dispatch._launch`.
OPS = {
    "complex_gemm": complex_gemm,
    "complex_gemm_bf16": complex_gemm_bf16,
    "complex_gemm_tn": complex_gemm_tn,
    "complex_gemm_t2d": complex_gemm_t2d,
    "complex_linear": complex_linear,
    "complex_mul": complex_mul,
    "butterfly_stage": butterfly_stage,
    "butterfly_stage_kahan": butterfly_stage_kahan,
    "stockham_radix4_stage": stockham_radix4_stage,
    "stockham_radix8_w8": stockham_radix8_w8,
    "stockham_radix8_w8_t2d": stockham_radix8_w8_t2d,
    "four_step_fft": four_step_fft,
    "fft": fft,
}
