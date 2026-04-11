"""Test complex neural network layers."""

import pytest
import torch
import numpy as np
from trnfft import ComplexTensor
from trnfft.nn import ComplexLinear, ComplexConv1d, ComplexBatchNorm1d, ComplexModReLU


class TestComplexLinear:

    def test_output_shape(self):
        layer = ComplexLinear(16, 32)
        x = ComplexTensor(torch.randn(4, 16), torch.randn(4, 16))
        y = layer(x)
        assert y.shape == (4, 32)

    def test_zero_imag_input(self):
        layer = ComplexLinear(8, 8, bias=False)
        x = ComplexTensor(torch.randn(2, 8))
        y = layer(x)
        assert y.shape == (2, 8)
        assert torch.all(torch.isfinite(y.real))
        assert torch.all(torch.isfinite(y.imag))


class TestComplexConv1d:

    def test_output_shape(self):
        layer = ComplexConv1d(1, 4, kernel_size=3, padding=1)
        x = ComplexTensor(torch.randn(2, 1, 100), torch.randn(2, 1, 100))
        y = layer(x)
        assert y.shape == (2, 4, 100)


class TestComplexBatchNorm1d:

    def test_output_shape(self):
        layer = ComplexBatchNorm1d(8)
        x = ComplexTensor(torch.randn(4, 8), torch.randn(4, 8))
        layer.train()
        y = layer(x)
        assert y.shape == (4, 8)


class TestComplexModReLU:

    def test_preserves_phase(self):
        layer = ComplexModReLU(1)
        layer.bias.data.fill_(0.0)
        x = ComplexTensor(torch.tensor([[3.0]]), torch.tensor([[4.0]]))
        y = layer(x)
        # Phase should be preserved: atan2(4,3)
        np.testing.assert_allclose(y.angle().item(), x.angle().item(), atol=1e-5)

    def test_thresholding(self):
        layer = ComplexModReLU(1)
        layer.bias.data.fill_(-10.0)  # Large negative bias kills small magnitudes
        x = ComplexTensor(torch.tensor([[0.1]]), torch.tensor([[0.1]]))
        y = layer(x)
        # Magnitude + bias < 0, so ReLU should zero it
        np.testing.assert_allclose(y.abs().item(), 0.0, atol=1e-6)


class TestGradients:
    """Verify autograd flows through ComplexTensor operations."""

    def test_complex_linear_gradient(self):
        layer = ComplexLinear(4, 4)
        x = ComplexTensor(torch.randn(2, 4, requires_grad=True),
                          torch.randn(2, 4, requires_grad=True))
        y = layer(x)
        loss = (y.real ** 2 + y.imag ** 2).sum()
        loss.backward()
        # Gradients should exist on all parameters
        assert layer.W_re.weight.grad is not None
        assert layer.W_im.weight.grad is not None
        assert x.real.grad is not None
        assert x.imag.grad is not None

    def test_fft_gradient(self):
        import trnfft
        x_re = torch.randn(16, requires_grad=True)
        x = ComplexTensor(x_re)
        X = trnfft.fft(x)
        loss = (X.real ** 2 + X.imag ** 2).sum()
        loss.backward()
        assert x_re.grad is not None
        assert torch.all(torch.isfinite(x_re.grad))

    def test_stft_gradient(self):
        import trnfft
        signal = torch.randn(512, requires_grad=True)
        S = trnfft.stft(signal, n_fft=64, hop_length=32, center=False)
        loss = S.abs().sum()
        loss.backward()
        assert signal.grad is not None
        assert torch.all(torch.isfinite(signal.grad))
