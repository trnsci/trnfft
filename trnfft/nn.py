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
        # (W_re + iW_im)(x_re + ix_im) = (W_re*x_re - W_im*x_im) + i(W_re*x_im + W_im*x_re)
        from .nki.dispatch import _use_nki

        if _use_nki():
            # NKI fused 4-real-matmul kernel reuses x tile across phases.
            from .nki.dispatch import complex_linear

            y = complex_linear(x, self.W_re.weight, self.W_im.weight)
            re, im = y.real, y.imag
        else:
            re = self.W_re(x.real) - self.W_im(x.imag)
            im = self.W_re(x.imag) + self.W_im(x.real)
        if self.bias_re is not None:
            re = re + self.bias_re
            im = im + self.bias_im
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
        re = self.conv_re(x.real) - self.conv_im(x.imag)
        im = self.conv_re(x.imag) + self.conv_im(x.real)
        if self.bias_re is not None:
            re = re + self.bias_re.unsqueeze(-1)
            im = im + self.bias_im.unsqueeze(-1)
        return ComplexTensor(re, im)


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
        # Add bias (broadcast over non-feature dims)
        bias = self.bias
        while bias.dim() < mag.dim():
            bias = bias.unsqueeze(0)
        if mag.dim() > bias.dim():
            bias = bias.unsqueeze(-1)

        activated_mag = torch.relu(mag + bias)
        # Avoid division by zero
        safe_mag = torch.clamp(mag, min=1e-8)
        scale = activated_mag / safe_mag
        return ComplexTensor(x.real * scale, x.imag * scale)
