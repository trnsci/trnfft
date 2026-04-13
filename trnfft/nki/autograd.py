"""autograd.Function wrappers for NKI kernels.

NKI kernels (``@nki.jit`` functions) return tensors allocated in
``nl.shared_hbm`` that have no ``grad_fn`` — calling one inside a
training loop silently detaches the autograd graph and subsequent
``loss.backward()`` raises ``RuntimeError: element 0 of tensors does
not require grad and does not have a grad_fn``.

This module wraps each kernel-driven dispatcher (complex GEMM, complex
Linear, complex elementwise multiply, FFT) in a ``torch.autograd.Function``
whose ``backward`` emits the analytic adjoint. Forward runs on NKI;
backward uses plain PyTorch ops on the same device for simplicity.
Backward-on-NKI is a possible future optimization (#56 follow-up).

Adjoints (all for complex operators; storage is split-real/imag):

* complex elementwise multiply  c = a * b
      da = dc * conj(b)
      db = dc * conj(a)

* complex GEMM                  C = A @ B
      dA = dC @ conj(B)^T
      dB = conj(A)^T @ dC

* complex linear                Y = X @ W^T        (W is (K_out, K_in))
      dX = dY @ conj(W)
      dW = dY^T @ conj(X)

* FFT                           y = fft(x, n)   (unnormalized forward)
      dx = ifft(dy) * n              -- our ifft divides by n; re-scale
* IFFT                          y = fft(x, n, inverse=True)
      dx = fft(dy) / n
"""

from __future__ import annotations

import torch

# NOTE: These Functions import kernel helpers at call time to avoid circular
# imports with dispatch.py and fft_core.py.


# ---------------------------------------------------------------------------
# Elementwise complex multiply
# ---------------------------------------------------------------------------

class _ComplexMulFn(torch.autograd.Function):
    """c = a * b (elementwise complex) with autograd support."""

    @staticmethod
    def forward(ctx, a_real, a_imag, b_real, b_imag):
        from .dispatch import _to_xla, _complex_mul_kernel
        (ar, ai, br, bi), orig = _to_xla(a_real, a_imag, b_real, b_imag)
        c_real, c_imag = _complex_mul_kernel(ar, ai, br, bi)
        ctx.save_for_backward(a_real, a_imag, b_real, b_imag)
        return c_real.to(orig), c_imag.to(orig)

    @staticmethod
    def backward(ctx, grad_c_real, grad_c_imag):
        a_real, a_imag, b_real, b_imag = ctx.saved_tensors
        # dA = dC * conj(B); dB = dC * conj(A)
        dA_real = grad_c_real * b_real + grad_c_imag * b_imag
        dA_imag = grad_c_imag * b_real - grad_c_real * b_imag
        dB_real = grad_c_real * a_real + grad_c_imag * a_imag
        dB_imag = grad_c_imag * a_real - grad_c_real * a_imag
        return dA_real, dA_imag, dB_real, dB_imag


def complex_mul_autograd(a_real, a_imag, b_real, b_imag):
    """Autograd-aware front to _complex_mul_kernel. Returns (c_real, c_imag)."""
    return _ComplexMulFn.apply(a_real, a_imag, b_real, b_imag)


# ---------------------------------------------------------------------------
# Complex GEMM
# ---------------------------------------------------------------------------

class _ComplexGEMMFn(torch.autograd.Function):
    """C = A @ B (complex GEMM) with autograd support."""

    @staticmethod
    def forward(ctx, a_real, a_imag, b_real, b_imag):
        from .dispatch import _to_xla, _complex_gemm_kernel
        (ar, ai, br, bi), orig = _to_xla(a_real, a_imag, b_real, b_imag)
        c_real, c_imag = _complex_gemm_kernel(ar, ai, br, bi)
        ctx.save_for_backward(a_real, a_imag, b_real, b_imag)
        return c_real.to(orig), c_imag.to(orig)

    @staticmethod
    def backward(ctx, grad_c_real, grad_c_imag):
        a_real, a_imag, b_real, b_imag = ctx.saved_tensors
        # dA = dC @ conj(B)^T = dC @ B^H
        # For split real/imag:
        #   (dCr + i dCi) @ (Br - i Bi)^T = (dCr Br^T + dCi Bi^T) + i (dCi Br^T - dCr Bi^T)
        dA_real = grad_c_real @ b_real.transpose(-2, -1) + grad_c_imag @ b_imag.transpose(-2, -1)
        dA_imag = grad_c_imag @ b_real.transpose(-2, -1) - grad_c_real @ b_imag.transpose(-2, -1)
        # dB = conj(A)^T @ dC = A^H @ dC
        #   (Ar^T - i Ai^T)(dCr + i dCi) = (Ar^T dCr + Ai^T dCi) + i (Ar^T dCi - Ai^T dCr)
        dB_real = a_real.transpose(-2, -1) @ grad_c_real + a_imag.transpose(-2, -1) @ grad_c_imag
        dB_imag = a_real.transpose(-2, -1) @ grad_c_imag - a_imag.transpose(-2, -1) @ grad_c_real
        return dA_real, dA_imag, dB_real, dB_imag


def complex_gemm_autograd(a_real, a_imag, b_real, b_imag):
    return _ComplexGEMMFn.apply(a_real, a_imag, b_real, b_imag)


# ---------------------------------------------------------------------------
# Complex Linear  Y = X @ W^T  (W is (K_out, K_in), as stored in nn.Linear)
# ---------------------------------------------------------------------------

class _ComplexLinearFn(torch.autograd.Function):
    """Y = X @ W^T (complex linear) with autograd support."""

    @staticmethod
    def forward(ctx, x_real, x_imag, w_real, w_imag):
        from .dispatch import _to_xla, _complex_linear_kernel
        (xr, xi, wr, wi), orig = _to_xla(x_real, x_imag, w_real, w_imag)
        y_real, y_imag = _complex_linear_kernel(xr, xi, wr, wi)
        ctx.save_for_backward(x_real, x_imag, w_real, w_imag)
        return y_real.to(orig), y_imag.to(orig)

    @staticmethod
    def backward(ctx, grad_y_real, grad_y_imag):
        x_real, x_imag, w_real, w_imag = ctx.saved_tensors
        # Y = X @ W^T
        # dX = dY @ conj(W)
        #   = (dYr + i dYi) @ (Wr - i Wi)
        dX_real = grad_y_real @ w_real + grad_y_imag @ w_imag
        dX_imag = grad_y_imag @ w_real - grad_y_real @ w_imag
        # dW = dY^T @ conj(X)
        #   = (dYr^T + i dYi^T) @ (Xr - i Xi)
        dW_real = grad_y_real.transpose(-2, -1) @ x_real + grad_y_imag.transpose(-2, -1) @ x_imag
        dW_imag = grad_y_imag.transpose(-2, -1) @ x_real - grad_y_real.transpose(-2, -1) @ x_imag
        return dX_real, dX_imag, dW_real, dW_imag


def complex_linear_autograd(x_real, x_imag, w_real, w_imag):
    return _ComplexLinearFn.apply(x_real, x_imag, w_real, w_imag)


# ---------------------------------------------------------------------------
# FFT (wraps the full _cooley_tukey_nki, not individual butterfly stages)
# ---------------------------------------------------------------------------

class _FFTFn(torch.autograd.Function):
    """Forward: y = fft(x) (NKI-accelerated butterfly).

    Backward for FFT:   dx = ifft(dy) * n
    Backward for IFFT:  dx = fft(dy) / n
    """

    @staticmethod
    def forward(ctx, x_real, x_imag, inverse: bool, precision: str = "fast"):
        # Call the raw (non-autograd) FFT path to avoid infinite recursion.
        from ..fft_core import _cooley_tukey_nki_nograd
        from ..complex import ComplexTensor
        x = ComplexTensor(x_real, x_imag)
        y = _cooley_tukey_nki_nograd(x, inverse=inverse, precision=precision)
        ctx.inverse = inverse
        ctx.n = x.shape[-1]
        ctx.precision = precision
        return y.real, y.imag

    @staticmethod
    def backward(ctx, grad_y_real, grad_y_imag):
        from ..fft_core import _cooley_tukey_nki_nograd
        from ..complex import ComplexTensor
        grad_y = ComplexTensor(grad_y_real, grad_y_imag)
        if ctx.inverse:
            # forward was IFFT; backward is FFT(grad) / n
            grad_x = _cooley_tukey_nki_nograd(grad_y, inverse=False, precision=ctx.precision)
            grad_x = grad_x * (1.0 / ctx.n)
        else:
            # forward was FFT; backward is IFFT(grad) * n (undoing ifft's 1/n)
            grad_x = _cooley_tukey_nki_nograd(grad_y, inverse=True, precision=ctx.precision)
            grad_x = grad_x * ctx.n
        return grad_x.real, grad_x.imag, None, None  # None for inverse flag and precision


def fft_autograd(x_real, x_imag, inverse: bool, precision: str = "fast"):
    return _FFTFn.apply(x_real, x_imag, inverse, precision)
