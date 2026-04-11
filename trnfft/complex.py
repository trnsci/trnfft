"""
Split real/imaginary complex tensor representation for Trainium.

Trainium has no native complex dtype. This module provides ComplexTensor,
which stores real and imaginary parts as separate tensors and implements
complex arithmetic via real-valued operations.

Originally from neuron-complex-ops, now folded into trnfft.
"""

from __future__ import annotations

import torch
from typing import Optional


class ComplexTensor:
    """Complex tensor stored as paired real/imaginary tensors."""

    __slots__ = ("real", "imag")

    def __init__(self, real: torch.Tensor, imag: Optional[torch.Tensor] = None):
        if imag is None:
            if torch.is_complex(real):
                self.real = real.real.contiguous()
                self.imag = real.imag.contiguous()
            else:
                self.real = real.contiguous()
                self.imag = torch.zeros_like(real)
        else:
            assert real.shape == imag.shape, f"Shape mismatch: {real.shape} vs {imag.shape}"
            assert real.dtype == imag.dtype, f"Dtype mismatch: {real.dtype} vs {imag.dtype}"
            self.real = real.contiguous()
            self.imag = imag.contiguous()

    @property
    def shape(self) -> torch.Size:
        return self.real.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.real.dtype

    @property
    def device(self) -> torch.device:
        return self.real.device

    @classmethod
    def from_polar(cls, magnitude: torch.Tensor, phase: torch.Tensor) -> ComplexTensor:
        return cls(magnitude * torch.cos(phase), magnitude * torch.sin(phase))

    def to_torch_complex(self) -> torch.Tensor:
        return torch.complex(self.real, self.imag)

    def abs(self) -> torch.Tensor:
        return torch.sqrt(self.real ** 2 + self.imag ** 2)

    def angle(self) -> torch.Tensor:
        return torch.atan2(self.imag, self.real)

    def conj(self) -> ComplexTensor:
        return ComplexTensor(self.real, -self.imag)

    def __add__(self, other):
        if isinstance(other, ComplexTensor):
            return ComplexTensor(self.real + other.real, self.imag + other.imag)
        if isinstance(other, (int, float)):
            return ComplexTensor(self.real + other, self.imag.clone())
        if isinstance(other, torch.Tensor) and not torch.is_complex(other):
            return ComplexTensor(self.real + other, self.imag.clone())
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ComplexTensor):
            return ComplexTensor(self.real - other.real, self.imag - other.imag)
        if isinstance(other, (int, float)):
            return ComplexTensor(self.real - other, self.imag.clone())
        if isinstance(other, torch.Tensor) and not torch.is_complex(other):
            return ComplexTensor(self.real - other, self.imag.clone())
        return NotImplemented

    def __mul__(self, other):
        """Element-wise complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
        if isinstance(other, ComplexTensor):
            re = self.real * other.real - self.imag * other.imag
            im = self.real * other.imag + self.imag * other.real
            return ComplexTensor(re, im)
        if isinstance(other, (int, float)):
            return ComplexTensor(self.real * other, self.imag * other)
        if isinstance(other, torch.Tensor) and not torch.is_complex(other):
            return ComplexTensor(self.real * other, self.imag * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        """Complex matrix multiply via 4 real matmuls."""
        if isinstance(other, ComplexTensor):
            return complex_matmul(self, other)
        return NotImplemented

    def __neg__(self) -> ComplexTensor:
        return ComplexTensor(-self.real, -self.imag)

    def __getitem__(self, key) -> ComplexTensor:
        return ComplexTensor(self.real[key], self.imag[key])

    def __setitem__(self, key, value: ComplexTensor):
        self.real[key] = value.real
        self.imag[key] = value.imag

    def __repr__(self) -> str:
        return f"ComplexTensor(shape={self.shape}, dtype={self.dtype})"

    def to(self, *args, **kwargs) -> ComplexTensor:
        return ComplexTensor(self.real.to(*args, **kwargs), self.imag.to(*args, **kwargs))

    def clone(self) -> ComplexTensor:
        return ComplexTensor(self.real.clone(), self.imag.clone())

    def reshape(self, *shape) -> ComplexTensor:
        return ComplexTensor(self.real.reshape(*shape), self.imag.reshape(*shape))

    def transpose(self, dim0: int, dim1: int) -> ComplexTensor:
        return ComplexTensor(
            self.real.transpose(dim0, dim1).contiguous(),
            self.imag.transpose(dim0, dim1).contiguous(),
        )

    def unsqueeze(self, dim: int) -> ComplexTensor:
        return ComplexTensor(self.real.unsqueeze(dim), self.imag.unsqueeze(dim))

    def squeeze(self, dim: Optional[int] = None) -> ComplexTensor:
        if dim is None:
            return ComplexTensor(self.real.squeeze(), self.imag.squeeze())
        return ComplexTensor(self.real.squeeze(dim), self.imag.squeeze(dim))


def complex_matmul(a: ComplexTensor, b: ComplexTensor) -> ComplexTensor:
    """Complex matrix multiply: C = A @ B using 4 real matmuls.

    C_real = A_real @ B_real - A_imag @ B_imag
    C_imag = A_real @ B_imag + A_imag @ B_real
    """
    re = torch.matmul(a.real, b.real) - torch.matmul(a.imag, b.imag)
    im = torch.matmul(a.real, b.imag) + torch.matmul(a.imag, b.real)
    return ComplexTensor(re, im)
