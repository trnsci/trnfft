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


class TestRFFTnD:
    """rfft2, irfft2, rfftn, irfftn — real-input N-D FFT variants."""

    def test_rfft2_vs_numpy(self):
        torch.manual_seed(42)
        for shape in [(8, 16), (16, 32)]:
            x = torch.randn(*shape)
            result = trnfft.rfft2(x)
            expected = np.fft.rfft2(x.numpy())
            assert result.shape == (shape[0], shape[1] // 2 + 1)
            np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_irfft2_roundtrip(self):
        torch.manual_seed(42)
        x = torch.randn(8, 16)
        X = trnfft.rfft2(x)
        recovered = trnfft.irfft2(X, s=x.shape)
        np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-4)

    def test_irfft2_default_shape(self):
        # With s=None, irfft2 infers last dim as 2*(N_half - 1). The second-
        # to-last dim is taken from the input shape directly.
        torch.manual_seed(42)
        x = torch.randn(8, 16)
        X = trnfft.rfft2(x)  # shape (8, 9)
        recovered = trnfft.irfft2(X)  # default: (8, 2*(9-1)) = (8, 16)
        assert recovered.shape == (8, 16)
        np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-4)

    def test_rfftn_vs_numpy_3d(self):
        torch.manual_seed(42)
        x = torch.randn(4, 8, 16)
        result = trnfft.rfftn(x)
        expected = np.fft.rfftn(x.numpy())
        assert result.shape == (4, 8, 16 // 2 + 1)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_irfftn_roundtrip_3d(self):
        torch.manual_seed(42)
        x = torch.randn(4, 8, 16)
        X = trnfft.rfftn(x)
        recovered = trnfft.irfftn(X, s=x.shape)
        np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-4)

    def test_rfft2_with_s_param(self):
        # s resizes the input before transform (zero-pad or truncate).
        torch.manual_seed(42)
        x = torch.randn(8, 16)
        padded = trnfft.rfft2(x, s=(16, 32))
        expected = np.fft.rfft2(x.numpy(), s=(16, 32))
        assert padded.shape == (16, 32 // 2 + 1)
        np.testing.assert_allclose(padded.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(padded.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)


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


class TestBluesteinsPrecision:
    """Precision modes for Bluestein (non-power-of-2 FFT).

    Empirical findings from the host path (CPU):

    * ``"fast"`` is the baseline.
    * ``"kahan"`` on the host only compensates the chirp multiplies — it
      matches "fast" because the error budget is dominated by the 3-FFT
      chain, not the chirps. The NKI butterfly Kahan variant is where
      this mode pays off (validated separately via ``test_fft_nki_kahan``).
    * ``"double"`` promotes the Bluestein host math to FP64. Bluestein
      sizes see ~10 orders of magnitude reduction in error; power-of-2
      sizes (which skip Bluestein entirely) are unaffected.
    """

    @pytest.mark.parametrize("n", [500, 1000])
    def test_fast_mode(self, n):
        """At N in [500, 1000], 'fast' clears the documented 2e-2 tolerance.
        Larger Bluestein sizes (N >> 1000) have higher fast-mode error —
        use 'double' for those. See TestBluesteinsPrecision docstring."""
        trnfft.set_precision("fast")
        try:
            torch.manual_seed(42)
            x = torch.randn(n)
            result = trnfft.fft(x)
            expected = np.fft.fft(x.numpy())
            np.testing.assert_allclose(result.real.numpy(), expected.real, atol=2e-2, rtol=2e-2)
            np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=2e-2, rtol=2e-2)
        finally:
            trnfft.set_precision("fast")

    @pytest.mark.parametrize("n", [500, 1000, 4097, 8193])
    def test_double_mode(self, n):
        """FP64 promotion for Bluestein gives ~1e-6 or better across all sizes."""
        trnfft.set_precision("double")
        try:
            torch.manual_seed(42)
            x = torch.randn(n)
            result = trnfft.fft(x)
            expected = np.fft.fft(x.numpy())
            # Empirically the rel error is 1e-11 or better; 1e-5 gives safety.
            np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-5, rtol=1e-5)
        finally:
            trnfft.set_precision("fast")

    @pytest.mark.parametrize("n", [4097, 8193])
    def test_double_beats_fast_at_large_n(self, n):
        """Confirm the user-visible benefit of 'double' over 'fast' at the
        sizes where fast falls apart. Fast's abs error at N=8193 can reach
        ~0.5; double keeps it below 1e-7."""
        torch.manual_seed(42)
        x = torch.randn(n)
        expected = np.fft.fft(x.numpy())

        trnfft.set_precision("fast")
        r_fast = trnfft.fft(x)
        err_fast = float(np.max(np.abs(r_fast.real.numpy() - expected.real) + np.abs(r_fast.imag.numpy() - expected.imag)))

        trnfft.set_precision("double")
        r_dbl = trnfft.fft(x)
        err_dbl = float(np.max(np.abs(r_dbl.real.numpy() - expected.real) + np.abs(r_dbl.imag.numpy() - expected.imag)))

        trnfft.set_precision("fast")
        assert err_dbl < err_fast / 100.0, (
            f"double ({err_dbl:.3e}) should be >=100x better than fast ({err_fast:.3e}) at N={n}"
        )

    def test_precision_getter_setter(self):
        import trnfft
        assert trnfft.get_precision() == "fast"
        trnfft.set_precision("kahan")
        assert trnfft.get_precision() == "kahan"
        trnfft.set_precision("double")
        assert trnfft.get_precision() == "double"
        trnfft.set_precision("fast")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="precision"):
            trnfft.set_precision("quad")


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


@pytest.mark.neuron
class TestNKIFFT:

    def test_fft_nki_vs_numpy(self, nki_backend):
        # Sizes >= 2048 exercise multi-partition tiling in butterfly kernel
        # (num_groups > PMAX=128 in stage 0).
        for n in [16, 64, 256, 1024, 4096, 16384]:
            torch.manual_seed(42)
            x = torch.randn(n)
            result = trnfft.fft(x)
            expected = np.fft.fft(x.numpy())
            np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
            np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)

    def test_fft_nki_roundtrip(self, nki_backend):
        for n in [16, 64, 256, 1024]:
            torch.manual_seed(42)
            x = torch.randn(n)
            X = trnfft.fft(x)
            recovered = trnfft.ifft(X)
            np.testing.assert_allclose(recovered.real.numpy(), x.numpy(), atol=1e-3)
            np.testing.assert_allclose(recovered.imag.numpy(), np.zeros(n), atol=1e-3)

    def test_fft2_nki_vs_numpy(self, nki_backend):
        for shape in [(16, 16), (64, 64)]:
            torch.manual_seed(42)
            x = torch.randn(*shape)
            result = trnfft.fft2(x)
            expected = np.fft.fft2(x.numpy())
            np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
            np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)

    def test_fftn_nki_vs_numpy(self, nki_backend):
        torch.manual_seed(42)
        x = torch.randn(4, 8, 8)
        result = trnfft.fftn(x)
        expected = np.fft.fftn(x.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)

    def test_batched_fft_nki(self, nki_backend):
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        result = trnfft.fft(x)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)

    def test_bluestein_nki_vs_numpy(self, nki_backend):
        # Bluestein wraps three power-of-2 FFTs that route through NKI butterfly,
        # while chirp multiply / padding / filter remain on host.
        for n in [7, 13, 100, 127]:
            torch.manual_seed(42)
            x = torch.randn(n)
            result = trnfft.fft(x)
            expected = np.fft.fft(x.numpy())
            tol = 1e-3 if n < 50 else 1e-2
            np.testing.assert_allclose(result.real.numpy(), expected.real, atol=tol, rtol=tol)
            np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=tol, rtol=tol)
