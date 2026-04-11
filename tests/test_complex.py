"""Test ComplexTensor arithmetic."""

import pytest
import torch
import numpy as np
from trnfft import ComplexTensor, complex_matmul


class TestComplexArithmetic:

    def test_from_real(self):
        x = ComplexTensor(torch.tensor([1.0, 2.0, 3.0]))
        assert x.imag.sum().item() == 0.0

    def test_from_torch_complex(self):
        z = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        x = ComplexTensor(z)
        np.testing.assert_allclose(x.real.numpy(), [1.0, 2.0])
        np.testing.assert_allclose(x.imag.numpy(), [3.0, 4.0])

    def test_add(self):
        a = ComplexTensor(torch.tensor([1.0]), torch.tensor([2.0]))
        b = ComplexTensor(torch.tensor([3.0]), torch.tensor([4.0]))
        c = a + b
        assert c.real.item() == 4.0
        assert c.imag.item() == 6.0

    def test_sub(self):
        a = ComplexTensor(torch.tensor([5.0]), torch.tensor([7.0]))
        b = ComplexTensor(torch.tensor([2.0]), torch.tensor([3.0]))
        c = a - b
        assert c.real.item() == 3.0
        assert c.imag.item() == 4.0

    def test_mul(self):
        # (1+2i)(3+4i) = -5+10i
        a = ComplexTensor(torch.tensor([1.0]), torch.tensor([2.0]))
        b = ComplexTensor(torch.tensor([3.0]), torch.tensor([4.0]))
        c = a * b
        np.testing.assert_allclose(c.real.item(), -5.0, atol=1e-6)
        np.testing.assert_allclose(c.imag.item(), 10.0, atol=1e-6)

    def test_mul_scalar(self):
        a = ComplexTensor(torch.tensor([2.0]), torch.tensor([3.0]))
        c = a * 2.0
        assert c.real.item() == 4.0
        assert c.imag.item() == 6.0

    def test_conj(self):
        a = ComplexTensor(torch.tensor([1.0]), torch.tensor([2.0]))
        assert a.conj().imag.item() == -2.0

    def test_abs(self):
        a = ComplexTensor(torch.tensor([3.0]), torch.tensor([4.0]))
        np.testing.assert_allclose(a.abs().item(), 5.0, atol=1e-6)

    def test_angle(self):
        a = ComplexTensor(torch.tensor([1.0]), torch.tensor([1.0]))
        np.testing.assert_allclose(a.angle().item(), np.pi / 4, atol=1e-6)

    def test_from_polar(self):
        c = ComplexTensor.from_polar(torch.tensor([1.0]), torch.tensor([np.pi / 4]))
        np.testing.assert_allclose(c.real.item(), np.cos(np.pi / 4), atol=1e-6)
        np.testing.assert_allclose(c.imag.item(), np.sin(np.pi / 4), atol=1e-6)

    def test_torch_roundtrip(self):
        a = ComplexTensor(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        b = ComplexTensor(a.to_torch_complex())
        np.testing.assert_allclose(b.real.numpy(), a.real.numpy())
        np.testing.assert_allclose(b.imag.numpy(), a.imag.numpy())

    def test_neg(self):
        a = ComplexTensor(torch.tensor([1.0]), torch.tensor([2.0]))
        b = -a
        assert b.real.item() == -1.0
        assert b.imag.item() == -2.0


class TestComplexMatmul:

    def test_identity(self):
        I = ComplexTensor(torch.eye(2), torch.zeros(2, 2))
        b = ComplexTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                          torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        c = complex_matmul(I, b)
        np.testing.assert_allclose(c.real.numpy(), b.real.numpy(), atol=1e-5)
        np.testing.assert_allclose(c.imag.numpy(), b.imag.numpy(), atol=1e-5)

    def test_vs_numpy(self):
        rng = np.random.default_rng(42)
        a_np = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        b_np = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        expected = a_np @ b_np

        a = ComplexTensor(torch.tensor(a_np.real, dtype=torch.float32),
                          torch.tensor(a_np.imag, dtype=torch.float32))
        b = ComplexTensor(torch.tensor(b_np.real, dtype=torch.float32),
                          torch.tensor(b_np.imag, dtype=torch.float32))
        c = complex_matmul(a, b)

        np.testing.assert_allclose(c.real.numpy(), expected.real, atol=1e-4)
        np.testing.assert_allclose(c.imag.numpy(), expected.imag, atol=1e-4)

    def test_matmul_operator(self):
        a = ComplexTensor(torch.eye(3), torch.zeros(3, 3))
        b = ComplexTensor(torch.ones(3, 3), torch.ones(3, 3))
        c = a @ b
        np.testing.assert_allclose(c.real.numpy(), b.real.numpy(), atol=1e-5)
