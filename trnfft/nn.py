"""
Complex-valued neural network layers for Trainium.

Folded from neuron-complex-ops. These layers operate on ComplexTensor
and are used in speech enhancement, physics-informed NNs, etc.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .complex import ComplexTensor
from .nki.dispatch import _fp64_host_guard


class ComplexLinear(nn.Module):
    """Complex-valued linear layer: y = Wx + b (complex arithmetic)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.W_re = nn.Linear(in_features, out_features, bias=False)
        self.W_im = nn.Linear(in_features, out_features, bias=False)
        self.bias_re = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.bias_im = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_re.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_im.weight, a=math.sqrt(5))

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        return _complex_linear_forward(
            x, self.W_re.weight, self.W_im.weight, self.bias_re, self.bias_im
        )


@_fp64_host_guard("ComplexLinear")
def _complex_linear_forward(x, w_re, w_im, b_re, b_im) -> ComplexTensor:
    """``ComplexLinear.forward`` on its parameters, so the float64 host guard
    (``dispatch._fp64_host_guard``) sees the weights as operands."""
    # (W_re + iW_im)(x_re + ix_im) = (W_re*x_re - W_im*x_im) + i(W_re*x_im + W_im*x_re)
    from .nki.dispatch import _use_nki

    if _use_nki():
        # NKI fused 4-real-matmul kernel reuses x tile across phases.
        from .nki.dispatch import complex_linear

        y = complex_linear(x, w_re, w_im)
        re, im = y.real, y.imag
    else:
        lin = torch.nn.functional.linear
        re = lin(x.real, w_re) - lin(x.imag, w_im)
        im = lin(x.imag, w_re) + lin(x.real, w_im)
    if b_re is not None:
        re = re + b_re
        im = im + b_im
    return ComplexTensor(re, im)


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv_re = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.conv_im = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        if bias:
            self.bias_re = nn.Parameter(torch.zeros(out_channels))
            self.bias_im = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_re = None
            self.bias_im = None

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        return _complex_conv1d_forward(
            x,
            self.conv_re.weight,
            self.conv_im.weight,
            self.bias_re,
            self.bias_im,
            self.conv_re.stride,
            self.conv_re.padding,
        )


@_fp64_host_guard("ComplexConv1d")
def _complex_conv1d_forward(x, w_re, w_im, b_re, b_im, stride, padding) -> ComplexTensor:
    """``ComplexConv1d.forward`` on its parameters, so the float64 host guard
    (``dispatch._fp64_host_guard``) sees the weights as operands (#134).

    Trainium has no fp64 datapath: a float64 ``aten::convolution`` on the
    neuron device fails to compile. Under backend "auto" the guard computes it
    on the host in float64. float64 that still reaches here on the device
    (backend "nki", or ``TRNFFT_FP64_ON_DEVICE=1``) is convolved in float32 on
    the device and returned as float64, the device's precision either way;
    "nki" warns once, ``TRNFFT_REQUIRE_NKI=1`` raises.
    """
    from .nki import dispatch

    operands = [t for t in (x.real, x.imag, w_re, w_im, b_re, b_im) if t is not None]
    out_dtype = None
    on_host_route = getattr(dispatch._FP64_HOST_ROUTE, "on", False)
    if (
        dispatch._backend != "pytorch"
        and not on_host_route
        and any(t.dtype == torch.float64 and dispatch._on_neuron_device(t) for t in operands)
    ):
        _conv1d_fp64_on_device()
        out_dtype = torch.float64

        def f32(t):
            return None if t is None else t.float()

        x = ComplexTensor(x.real.float(), x.imag.float())
        w_re, w_im, b_re, b_im = f32(w_re), f32(w_im), f32(b_re), f32(b_im)

    conv = torch.nn.functional.conv1d
    re = conv(x.real, w_re, None, stride, padding) - conv(x.imag, w_im, None, stride, padding)
    im = conv(x.imag, w_re, None, stride, padding) + conv(x.real, w_im, None, stride, padding)
    if b_re is not None:
        re = re + b_re.unsqueeze(-1)
        im = im + b_im.unsqueeze(-1)
    if out_dtype is not None:
        re, im = re.to(out_dtype), im.to(out_dtype)
    return ComplexTensor(re, im)


def _conv1d_fp64_on_device() -> None:
    """Policy for float64 ``ComplexConv1d`` operands left on the neuron device."""
    import warnings

    from .nki import dispatch

    if dispatch._require_nki():
        raise NotImplementedError(
            "trnfft ComplexConv1d: float64 on the neuron device has no kernel and no "
            "device fp64 convolution. Unset TRNFFT_REQUIRE_NKI to compute it on the "
            "host (backend 'auto') or in float32 on the device (backend 'nki')."
        )
    if dispatch._backend == "nki" and "ComplexConv1d" not in dispatch._FP64_WARNED:
        dispatch._FP64_WARNED.add("ComplexConv1d")
        warnings.warn(
            "trnfft ComplexConv1d: float64 on the neuron device under backend 'nki' "
            "is convolved in float32 on the device, so precision is reduced to "
            "float32 (the result is float64). Use backend 'auto' for float64 "
            "accuracy on the host. Warned once per op.",
            UserWarning,
            stacklevel=4,
        )


class ComplexBatchNorm1d(nn.Module):
    """Batch normalization for complex tensors.

    Uses independent normalization of real and imaginary parts (simpler variant).
    The covariance-based alternative (Trabelsi et al. 2018) jointly normalizes
    using the 2x2 covariance matrix — more principled but heavier. This simpler
    form works well in practice for speech enhancement and cIRM estimation.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.bn_re = nn.BatchNorm1d(num_features, eps=eps)
        self.bn_im = nn.BatchNorm1d(num_features, eps=eps)

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        return ComplexTensor(self.bn_re(x.real), self.bn_im(x.imag))


class ComplexModReLU(nn.Module):
    """Modulus ReLU: f(z) = ReLU(|z| + b) * z / |z|

    Applies ReLU to magnitude while preserving phase.
    Learnable bias b allows the network to learn a threshold.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        mag = x.abs()
        # Broadcast the (F,) bias over the non-feature dims. The feature dim is
        # dim 1 for rank >= 2, as in ComplexConv1d / BatchNorm1d: (B, F) or
        # (B, F, L, ...). Unsqueezing at dim 0 only put it on the *last* dim,
        # wrong for (B, F, L) inputs (#115).
        bias = self.bias
        if mag.dim() >= 2:
            bias = bias.reshape(1, -1, *([1] * (mag.dim() - 2)))

        activated_mag = torch.relu(mag + bias)
        # Avoid division by zero
        safe_mag = torch.clamp(mag, min=1e-8)
        scale = activated_mag / safe_mag
        return ComplexTensor(x.real * scale, x.imag * scale)
