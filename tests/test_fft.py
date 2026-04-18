"""Test 1D FFT correctness against numpy.fft."""

import numpy as np
import pytest
import torch

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
        err_fast = float(
            np.max(
                np.abs(r_fast.real.numpy() - expected.real)
                + np.abs(r_fast.imag.numpy() - expected.imag)
            )
        )

        trnfft.set_precision("double")
        r_dbl = trnfft.fft(x)
        err_dbl = float(
            np.max(
                np.abs(r_dbl.real.numpy() - expected.real)
                + np.abs(r_dbl.imag.numpy() - expected.imag)
            )
        )

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


class TestDFTGEMM:
    """Direct unit tests for the DFT-as-GEMM fast path.

    The path dispatches automatically when running under NKI backend
    (see _cooley_tukey_nki_nograd in fft_core.py), but the math is
    equally valid on CPU — _fft_via_gemm calls through complex_gemm,
    which falls back to torch.matmul without NKI.
    """

    @pytest.mark.parametrize("n", [8, 16, 32, 64, 128])
    def test_matches_numpy(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm

        torch.manual_seed(42)
        x = torch.randn(n)
        result = _fft_via_gemm(ComplexTensor(x, torch.zeros(n)), inverse=False)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("n", [8, 32, 128])
    def test_roundtrip(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        X = _fft_via_gemm(ct, inverse=False)
        back = _fft_via_gemm(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.numpy(), atol=1e-3)
        np.testing.assert_allclose(back.imag.numpy(), np.zeros(n), atol=1e-3)

    def test_batched(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm

        torch.manual_seed(42)
        x = torch.randn(4, 64)
        result = _fft_via_gemm(ComplexTensor(x, torch.zeros_like(x)), inverse=False)
        expected = np.fft.fft(x.numpy(), axis=-1)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)


class TestDFTGEMMDouble:
    """Tests for the FP64 CPU DFT-GEMM path (_fft_via_gemm_double).

    This path activates when precision="double" and n <= _DOUBLE_GEMM_THRESHOLD.
    It bypasses NKI and computes W @ x in float64 on CPU, achieving ~1e-14
    relative error vs numpy reference.

    These tests run on CPU without NKI — they call _fft_via_gemm_double directly
    and also verify that the precision="double" dispatch wires up correctly.
    """

    @pytest.mark.parametrize("n", [8, 64, 256, 512, 1024])
    def test_matches_numpy_double(self, n):
        # Output dtype matches FP32 input (consistent with Bluestein "double" behaviour).
        # FP32 quantization floor is ~1e-7; tolerance 1e-6 gives comfortable margin.
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm_double

        torch.manual_seed(42)
        x = torch.randn(n)
        result = _fft_via_gemm_double(ComplexTensor(x, torch.zeros(n)), inverse=False)
        expected = np.fft.fft(x.numpy().astype(np.float64))
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-6, rtol=1e-6)

    def test_matches_numpy_double_fp64_input(self):
        # When input is already FP64 the output is FP64 and achieves ~1e-14 error.
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm_double

        torch.manual_seed(42)
        n = 512
        x = torch.randn(n, dtype=torch.float64)
        result = _fft_via_gemm_double(
            ComplexTensor(x, torch.zeros(n, dtype=torch.float64)), inverse=False
        )
        expected = np.fft.fft(x.numpy())
        assert result.real.dtype == torch.float64
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-10, rtol=1e-10)

    def test_roundtrip_double(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm_double

        torch.manual_seed(42)
        n = 512
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        X = _fft_via_gemm_double(ct, inverse=False)
        back = _fft_via_gemm_double(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.numpy(), atol=1e-5)
        np.testing.assert_allclose(back.imag.numpy(), np.zeros(n), atol=1e-5)

    def test_batched_double(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm_double

        torch.manual_seed(42)
        x = torch.randn(4, 256)
        result = _fft_via_gemm_double(ComplexTensor(x, torch.zeros_like(x)), inverse=False)
        expected = np.fft.fft(x.numpy().astype(np.float64), axis=-1)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-6, rtol=1e-6)

    def test_inverse_double_at_n512(self):
        """_fft_via_gemm_double correctly handles inverse=True at N=512."""
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm_double

        torch.manual_seed(42)
        n = 512
        x = torch.randn(n) + 1j * torch.randn(n)
        ct = ComplexTensor(x.real, x.imag)
        X = _fft_via_gemm_double(ct, inverse=False)
        back = _fft_via_gemm_double(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.real.numpy(), atol=1e-5)
        np.testing.assert_allclose(back.imag.numpy(), x.imag.numpy(), atol=1e-5)

    def test_double_beats_fast_precision_at_n256(self):
        """ "double" mode is more accurate than "fast" mode at N=256."""
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm, _fft_via_gemm_double

        torch.manual_seed(99)
        n = 256
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        expected = np.fft.fft(x.numpy().astype(np.float64))

        fast = _fft_via_gemm(ct, inverse=False)
        double = _fft_via_gemm_double(ct, inverse=False)

        err_fast = np.abs(fast.real.numpy() - expected.real).max()
        err_double = np.abs(double.real.numpy() - expected.real).max()
        assert err_double < err_fast, (
            f"double ({err_double:.2e}) should be more accurate than fast ({err_fast:.2e})"
        )
        assert err_double < 1e-5


class TestStockhamRadix4:
    """CPU reference for radix-4 Stockham FFT (trnfft.stockham).

    Validates indexing + twiddle math that the NKI port mirrors. The
    Stockham advantage over DFT-GEMM shows up directly in FP32 error
    scaling: log_4(N) stages of small 4x4 DFTs accumulate ~1e-6 relative
    error at N=4096, vs DFT-GEMM's ~2% at N=1024.
    """

    @pytest.mark.parametrize("n", [16, 64, 256, 1024, 4096])
    def test_matches_numpy(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix4

        torch.manual_seed(42)
        x = torch.randn(n)
        result = stockham_radix4(ComplexTensor(x, torch.zeros(n)), inverse=False)
        expected = np.fft.fft(x.numpy())
        # Stockham-FP32 is tighter than DFT-GEMM: 1e-4 works across the range.
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("n", [16, 64, 256, 1024, 4096])
    def test_roundtrip(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix4

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        X = stockham_radix4(ct, inverse=False)
        back = stockham_radix4(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.numpy(), atol=1e-5)
        np.testing.assert_allclose(back.imag.numpy(), np.zeros(n), atol=1e-5)

    def test_ifft_vs_numpy(self):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix4

        torch.manual_seed(42)
        xr = torch.randn(256)
        xi = torch.randn(256)
        result = stockham_radix4(ComplexTensor(xr, xi), inverse=True)
        expected = np.fft.ifft(xr.numpy() + 1j * xi.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_batched(self):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix4

        torch.manual_seed(42)
        x = torch.randn(4, 256)
        result = stockham_radix4(ComplexTensor(x, torch.zeros_like(x)), inverse=False)
        expected = np.fft.fft(x.numpy(), axis=-1)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_rejects_non_power_of_four(self):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix4

        with pytest.raises(AssertionError, match="N=4\\^k"):
            stockham_radix4(ComplexTensor(torch.randn(32), torch.zeros(32)))


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

    def test_kahan_butterfly_compiles_and_matches_fast(self, nki_backend):
        # Compile + sanity check for the Dekker-2Prod butterfly variant.
        # Kahan is a precision tightening, not a different answer — it should
        # agree with "fast" to ~FP32 rtol on typical inputs. On-silicon
        # precision characterization (does kahan actually help FP32?) is a
        # separate follow-up — here we just confirm it compiles and doesn't
        # diverge.
        #
        # Force butterfly on both sides by zeroing the DFT-GEMM threshold;
        # otherwise "fast" would route to DFT-GEMM (a different algorithm
        # with its own FP32 error profile) and this test would measure
        # algorithm divergence instead of the kahan-vs-fast butterfly
        # question it's supposed to answer.
        from trnfft import fft_core, get_precision, set_precision

        old_prec = get_precision()
        old_thr = fft_core._DFT_GEMM_THRESHOLD
        fft_core._DFT_GEMM_THRESHOLD = 0
        try:
            torch.manual_seed(42)
            x = torch.randn(1024)
            set_precision("fast")
            y_fast = trnfft.fft(x)
            set_precision("kahan")
            y_kahan = trnfft.fft(x)
            np.testing.assert_allclose(
                y_kahan.real.numpy(), y_fast.real.numpy(), atol=1e-3, rtol=1e-3
            )
            np.testing.assert_allclose(
                y_kahan.imag.numpy(), y_fast.imag.numpy(), atol=1e-3, rtol=1e-3
            )
        finally:
            set_precision(old_prec)
            fft_core._DFT_GEMM_THRESHOLD = old_thr
