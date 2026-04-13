"""Test complex neural network layers."""

import numpy as np
import pytest
import torch

from trnfft import ComplexTensor
from trnfft.nn import ComplexBatchNorm1d, ComplexConv1d, ComplexLinear, ComplexModReLU


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
        x = ComplexTensor(
            torch.randn(2, 4, requires_grad=True), torch.randn(2, 4, requires_grad=True)
        )
        y = layer(x)
        loss = (y.real**2 + y.imag**2).sum()
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
        loss = (X.real**2 + X.imag**2).sum()
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


@pytest.mark.neuron
class TestNKILayers:
    """Validate that NN layers work end-to-end with NKI dispatch."""

    def test_complex_linear_nki_vs_pytorch(self, nki_backend):
        from trnfft import set_backend

        torch.manual_seed(42)
        # Sizes: M=K_in=128, K_out=256. All multiples of 128 for clean tiling.
        layer = ComplexLinear(128, 256, bias=True)
        x = ComplexTensor(torch.randn(128, 128), torch.randn(128, 128))

        # NKI path (active because of nki_backend fixture)
        y_nki = layer(x)

        # PyTorch reference
        set_backend("pytorch")
        y_ref = layer(x)
        set_backend("nki")  # restore for fixture's teardown

        np.testing.assert_allclose(
            y_nki.real.detach().numpy(), y_ref.real.detach().numpy(), atol=1e-3, rtol=1e-3
        )
        np.testing.assert_allclose(
            y_nki.imag.detach().numpy(), y_ref.imag.detach().numpy(), atol=1e-3, rtol=1e-3
        )

    def test_complex_conv1d_pytorch_fallback(self, nki_backend):
        # No NKI conv kernel exists. Layer must still work via PyTorch ops.
        layer = ComplexConv1d(1, 4, kernel_size=3, padding=1)
        x = ComplexTensor(torch.randn(2, 1, 64), torch.randn(2, 1, 64))
        y = layer(x)
        assert y.shape == (2, 4, 64)
        assert torch.all(torch.isfinite(y.real))
        assert torch.all(torch.isfinite(y.imag))

    def test_complex_modrelu_pytorch_fallback(self, nki_backend):
        # No NKI ModReLU kernel exists. Falls back to elementwise PyTorch ops.
        layer = ComplexModReLU(8)
        x = ComplexTensor(torch.randn(4, 8), torch.randn(4, 8))
        y = layer(x)
        assert y.shape == (4, 8)
        assert torch.all(torch.isfinite(y.abs()))


@pytest.mark.neuron
class TestNKIGradients:
    """Autograd must flow through NKI kernels (#56).

    These tests exercise the ``torch.autograd.Function`` wrappers in
    ``trnfft/nki/autograd.py``. Before the fix, any of these would raise
    ``RuntimeError: element 0 of tensors does not require grad and does not
    have a grad_fn``.
    """

    def test_complex_linear_nki_grad(self, nki_backend):
        torch.manual_seed(42)
        layer = ComplexLinear(128, 256, bias=True)
        x = ComplexTensor(
            torch.randn(128, 128, requires_grad=True),
            torch.randn(128, 128, requires_grad=True),
        )
        y = layer(x)
        loss = (y.real**2 + y.imag**2).sum()
        loss.backward()
        assert layer.W_re.weight.grad is not None
        assert layer.W_im.weight.grad is not None
        assert torch.all(torch.isfinite(layer.W_re.weight.grad))
        assert torch.all(torch.isfinite(layer.W_im.weight.grad))
        assert x.real.grad is not None
        assert torch.all(torch.isfinite(x.real.grad))

    def test_fft_nki_grad(self, nki_backend):
        import trnfft

        torch.manual_seed(42)
        x_re = torch.randn(64, requires_grad=True)
        x = ComplexTensor(x_re)
        X = trnfft.fft(x)
        loss = (X.real**2 + X.imag**2).sum()
        loss.backward()
        assert x_re.grad is not None
        assert torch.all(torch.isfinite(x_re.grad))

    def test_stft_nki_grad(self, nki_backend):
        import trnfft

        torch.manual_seed(42)
        signal = torch.randn(2048, requires_grad=True)
        S = trnfft.stft(signal, n_fft=128, hop_length=64, center=False)
        loss = S.abs().sum()
        loss.backward()
        assert signal.grad is not None
        assert torch.all(torch.isfinite(signal.grad))
