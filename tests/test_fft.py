"""Test 1D FFT correctness against numpy.fft."""

import pytest
import torch
import numpy as np
import trnfft
from trnfft import ComplexTensor


class TestFFT1D:

    def test_impulse(self):
        n = 16
        x = torch.zeros(n)
        x[0] = 1.0
        result = trnfft.fft(x)
        np.testing.assert_allclose(result.real.numpy(), np.ones(n), atol=1e-6)
        np.testing.assert_allclose(result.imag.numpy(), np.zeros(n), atol=1e-6)

    def test_constant(self):
        n = 16
        x = torch.ones(n)
        result = trnfft.fft(x)
        np.testing.assert_allclose(result.real[0].item(), float(n), atol=1e-5)
        np.testing.assert_allclose(result.abs()[1:].numpy(), np.zeros(n - 1), atol=1e-5)

    def test_single_frequency(self):
        n = 64
        k = 5
        t = torch.arange(n, dtype=torch.float32)
        x = torch.cos(2 * np.pi * k * t / n)
        result = trnfft.fft(x)
        mags = result.abs()
        assert mags[k].item() > n / 2 - 1
        assert mags[n - k].item() > n / 2 - 1

    def test_vs_numpy_power_of_2(self, power_of_2_size, random_real_signal):
        n = power_of_2_size
        x = random_real_signal(n)
        result = trnfft.fft(x)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_vs_numpy_complex_input(self, power_of_2_size, random_complex_signal):
        n = power_of_2_size
        x = random_complex_signal(n)
        result = trnfft.fft(x)
        x_np = x.real.numpy() + 1j * x.imag.numpy()
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_batched(self, random_real_signal):
        n = 64
        batch = 8
        x = random_real_signal(n, batch=batch)
        result = trnfft.fft(x)
        for i in range(batch):
            single = trnfft.fft(x[i])
            np.testing.assert_allclose(result.real[i].numpy(), single.real.numpy(), atol=1e-5)
            np.testing.assert_allclose(result.imag[i].numpy(), single.imag.numpy(), atol=1e-5)

    def test_zero_padding(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = trnfft.fft(x, n=8)
        expected = np.fft.fft([1, 2, 3, 4, 0, 0, 0, 0])
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-5)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-5)


class TestIFFT1D:

    def test_roundtrip(self, power_of_2_size, random_real_signal):
        n = power_of_2_size
        x = random_real_signal(n)
        X = trnfft.fft(x)
        recovered = trnfft.ifft(X)
        np.testing.assert_allclose(recovered.real.numpy(), x.numpy(), atol=1e-4)
        np.testing.assert_allclose(recovered.imag.numpy(), np.zeros(n), atol=1e-4)

    def test_roundtrip_complex(self, power_of_2_size, random_complex_signal):
        n = power_of_2_size
        x = random_complex_signal(n)
        X = trnfft.fft(x)
        recovered = trnfft.ifft(X)
        np.testing.assert_allclose(recovered.real.numpy(), x.real.numpy(), atol=1e-4)
        np.testing.assert_allclose(recovered.imag.numpy(), x.imag.numpy(), atol=1e-4)


class TestRFFT:

    def test_vs_numpy(self, random_real_signal):
        n = 128
        x = random_real_signal(n)
        result = trnfft.rfft(x)
        expected = np.fft.rfft(x.numpy())
        assert result.shape[-1] == n // 2 + 1
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4)

    def test_irfft_roundtrip(self, random_real_signal):
        n = 128
        x = random_real_signal(n)
        X = trnfft.rfft(x)
        recovered = trnfft.irfft(X, n=n)
        np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-4)


class TestBluestein:

    def test_vs_numpy_arbitrary(self, arbitrary_size, random_real_signal):
        n = arbitrary_size
        x = random_real_signal(n)
        result = trnfft.fft(x)
        expected = np.fft.fft(x.numpy())
        # Larger sizes accumulate more FP32 error through Bluestein's 3-FFT chain
        tol = 1e-3 if n < 500 else 2e-2
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=tol, rtol=tol)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=tol, rtol=tol)

    def test_roundtrip_arbitrary(self, arbitrary_size, random_real_signal):
        n = arbitrary_size
        x = random_real_signal(n)
        X = trnfft.fft(x)
        recovered = trnfft.ifft(X)
        np.testing.assert_allclose(recovered.real.numpy(), x.numpy(), atol=1e-3)


class TestFFT2D:

    def test_vs_numpy(self):
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((8, 8)).astype(np.float32)
        x = torch.tensor(x_np)
        result = trnfft.fft2(x)
        expected = np.fft.fft2(x_np)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3)


class TestFFTnD:

    def test_fftn_3d(self):
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8, 8)).astype(np.float32)
        x = torch.tensor(x_np)
        result = trnfft.fftn(x)
        expected = np.fft.fftn(x_np)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3)

    def test_fftn_subset_dims(self):
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8, 16)).astype(np.float32)
        x = torch.tensor(x_np)
        # FFT along last two dims only (equivalent to fft2)
        result = trnfft.fftn(x, dim=(-2, -1))
        expected = np.fft.fftn(x_np, axes=(-2, -1))
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3)

    def test_ifftn_roundtrip(self):
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8, 8)).astype(np.float32)
        x = torch.tensor(x_np)
        X = trnfft.fftn(x)
        recovered = trnfft.ifftn(X)
        np.testing.assert_allclose(recovered.real.numpy(), x_np, atol=1e-3)
        np.testing.assert_allclose(recovered.imag.numpy(), np.zeros_like(x_np), atol=1e-3)

    def test_fftn_single_dim(self):
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        x = torch.tensor(x_np)
        # FFT along dim 0 only
        result = trnfft.fftn(x, dim=(0,))
        expected = np.fft.fftn(x_np, axes=(0,))
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3)
