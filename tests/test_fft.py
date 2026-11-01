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


class TestStockhamRadix8:
    """CPU reference for radix-8 Stockham FFT (Thread B).

    N=512 is the primary new coverage: currently routes to 9 butterfly stages;
    radix-8 uses 3 stages. N=4096 drops from 6 radix-4 stages to 4 stages.
    The NKI port uses nc_matmul (Tensor engine) for the W_8 DFT matvec.
    """

    @pytest.mark.parametrize("n", [8, 64, 512, 4096])
    def test_matches_numpy(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix8

        # W_8 has irrational entries (±√2/2 ± i√2/2) requiring real FP32 multiplications.
        # N=4096 (4 stages of 8×8 matmuls) accumulates slightly more error than radix-4.
        tol = 5e-4 if n >= 4096 else 1e-4
        torch.manual_seed(42)
        x = torch.randn(n)
        result = stockham_radix8(ComplexTensor(x, torch.zeros(n)), inverse=False)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=tol, rtol=tol)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=tol, rtol=tol)

    @pytest.mark.parametrize("n", [8, 64, 512])
    def test_roundtrip(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix8

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        X = stockham_radix8(ct, inverse=False)
        back = stockham_radix8(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.numpy(), atol=1e-5)
        np.testing.assert_allclose(back.imag.numpy(), np.zeros(n), atol=1e-5)

    def test_ifft_vs_numpy(self):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix8

        torch.manual_seed(42)
        xr = torch.randn(512)
        xi = torch.randn(512)
        result = stockham_radix8(ComplexTensor(xr, xi), inverse=True)
        expected = np.fft.ifft(xr.numpy() + 1j * xi.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_batched(self):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix8

        torch.manual_seed(42)
        x = torch.randn(4, 64)
        result = stockham_radix8(ComplexTensor(x, torch.zeros_like(x)), inverse=False)
        expected = np.fft.fft(x.numpy(), axis=-1)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_rejects_non_power_of_eight(self):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_radix8

        with pytest.raises(AssertionError, match="N=8\\^k"):
            stockham_radix8(ComplexTensor(torch.randn(32), torch.zeros(32)))


class TestStockhamMixed:
    """CPU reference for mixed-radix [8^a, 4^b] Stockham FFT (v0.16).

    Key coverage: N=1024 ([8,8,4,4] = 4 stages vs radix-4's 5) and
    N=2048 ([8,8,8,4] = 4 stages vs butterfly's 11 — new Stockham coverage).
    """

    @pytest.mark.parametrize("n", [64, 1024, 2048])
    def test_matches_numpy(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_mixed_radix

        tol = 5e-4 if n >= 2048 else 1e-4
        torch.manual_seed(42)
        x = torch.randn(n)
        result = stockham_mixed_radix(ComplexTensor(x, torch.zeros(n)), inverse=False)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=tol, rtol=tol)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=tol, rtol=tol)

    @pytest.mark.parametrize("n", [64, 1024, 2048])
    def test_roundtrip(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_mixed_radix

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        X = stockham_mixed_radix(ct, inverse=False)
        back = stockham_mixed_radix(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.numpy(), atol=1e-4)
        np.testing.assert_allclose(back.imag.numpy(), np.zeros(n), atol=1e-4)

    def test_ifft_n2048(self):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_mixed_radix

        torch.manual_seed(42)
        xr, xi = torch.randn(2048), torch.randn(2048)
        result = stockham_mixed_radix(ComplexTensor(xr, xi), inverse=True)
        expected = np.fft.ifft(xr.numpy() + 1j * xi.numpy())
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=5e-4, rtol=5e-4)

    def test_batched(self):
        from trnfft.complex import ComplexTensor
        from trnfft.stockham import stockham_mixed_radix

        torch.manual_seed(42)
        x = torch.randn(4, 1024)
        result = stockham_mixed_radix(ComplexTensor(x, torch.zeros_like(x)), inverse=False)
        expected = np.fft.fft(x.numpy(), axis=-1)
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)


class TestHFFT:
    """Tests for hfft and ihfft (Hermitian FFT pair, v0.16)."""

    def test_hfft_matches_numpy(self):
        import trnfft
        from trnfft.complex import ComplexTensor

        torch.manual_seed(42)
        n = 16
        # Build a Hermitian one-sided spectrum (length n//2+1 = 9)
        half = n // 2 + 1
        xr = torch.randn(half)
        xi = torch.randn(half)
        # DC and Nyquist must be real for strict Hermitian
        xi[0] = 0.0
        if n % 2 == 0:
            xi[-1] = 0.0
        ct = ComplexTensor(xr, xi)
        result = trnfft.hfft(ct, n=n)
        expected = np.fft.hfft(xr.numpy() + 1j * xi.numpy(), n=n)
        assert result.shape[-1] == n
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-4, rtol=1e-4)

    def test_ihfft_matches_numpy(self):
        import trnfft

        torch.manual_seed(42)
        n = 16
        x = torch.randn(n)
        result = trnfft.ihfft(x, n=n)
        expected = np.fft.ihfft(x.numpy(), n=n)
        assert result.shape[-1] == n // 2 + 1
        np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-4, rtol=1e-4)

    def test_hfft_ihfft_roundtrip(self):
        import trnfft

        torch.manual_seed(42)
        n = 32
        x = torch.randn(n)
        result = trnfft.hfft(trnfft.ihfft(x, n=n), n=n)
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-4)

    def test_hfft_batched(self):
        import trnfft
        from trnfft.complex import ComplexTensor

        torch.manual_seed(42)
        n, half = 32, 17
        xr = torch.randn(4, half)
        xi = torch.randn(4, half)
        xi[..., 0] = 0.0
        xi[..., -1] = 0.0
        ct = ComplexTensor(xr, xi)
        result = trnfft.hfft(ct, n=n)
        expected = np.fft.hfft(xr.numpy() + 1j * xi.numpy(), n=n)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-4, rtol=1e-4)


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


class TestBluesteinDeviceReturn:
    """Non-power-of-2 (Bluestein) FFT must keep every tensor on the caller's
    device through the NKI dispatch path (trnfft#80 follow-up).

    ``meta`` stands in for the Neuron device: like a native ``neuron`` tensor,
    a ``meta`` tensor refuses to mix with a CPU tensor ("Tensor on device cpu
    is not on the expected device meta"). A fake ``torch_xla`` and a stand-in
    GEMM kernel let the dispatch path run on CPU; the kernel also rejects
    any input that was not moved to the device first.
    """

    def _install_fakes(self, monkeypatch, device, compute):
        import sys
        import types

        from trnfft import fft_core as fc
        from trnfft.nki import dispatch

        fake_xla = types.ModuleType("torch_xla")
        fake_xla.device = lambda: torch.device(device)
        monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
        # butterfly.py imports the real ``nki`` at module load; the GEMM path
        # under test never calls these kernels.
        fake_bfly = types.ModuleType("trnfft.nki.butterfly")
        fake_bfly.butterfly_stage_kernel = None
        fake_bfly.butterfly_stage_kernel_kahan = None
        monkeypatch.setitem(sys.modules, "trnfft.nki.butterfly", fake_bfly)

        def fake_kernel(a_re, a_im, b_re, b_im):
            for t in (a_re, a_im, b_re, b_im):
                if t.device != torch.device(device):
                    raise RuntimeError(f"input tensor is on {t.device} device, expected {device}")
            if not compute:
                shape = (a_re.shape[0], b_re.shape[1])
                return torch.empty(shape, device=device), torch.empty(shape, device=device)
            c_re = a_re @ b_re - a_im @ b_im
            c_im = a_re @ b_im + a_im @ b_re
            return c_re, c_im

        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_complex_gemm_kernel", fake_kernel, raising=False)
        monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(fc, "_use_nki", lambda: True)
        # Route the padded power-of-2 sub-FFTs (up to m=2048 for n=1000)
        # through the DFT-GEMM wrapper, the path the fake kernel covers.
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 4096)
        monkeypatch.setattr(fc, "_FOUR_STEP_BATCH_FUSED", False)

    @pytest.mark.parametrize("n", [7, 127, 129, 1000])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_non_pow2_stays_on_caller_device(self, monkeypatch, n, inverse):
        self._install_fakes(monkeypatch, "meta", compute=False)
        x = ComplexTensor(torch.empty(n, device="meta"), torch.empty(n, device="meta"))
        y = trnfft.ifft(x) if inverse else trnfft.fft(x)
        assert y.real.device.type == "meta"
        assert y.imag.device.type == "meta"
        assert y.shape == (n,)

    @pytest.mark.parametrize("n", [7, 127, 1000])
    def test_non_pow2_matches_numpy_via_dispatch(self, monkeypatch, n):
        self._install_fakes(monkeypatch, "cpu", compute=True)
        x = torch.randn(n, generator=torch.Generator().manual_seed(n))
        y = trnfft.fft(x)
        ref = np.fft.fft(x.numpy().astype(np.float64))
        rel = np.linalg.norm(y.to_torch_complex().numpy() - ref) / np.linalg.norm(ref)
        assert rel < 1e-3


class TestFourStepDispatch:
    """Four-step FFT-as-two-GEMMs on the NKI dispatch path (trnfft#82).

    Fake ``torch_xla`` + stand-in kernels (same pattern as
    ``TestBluesteinDeviceReturn``) let the dispatch path run on CPU and count
    kernel launches. Stockham kernels raise, so any fall-through is caught.
    """

    def _install_fakes(self, monkeypatch, device="cpu", batch_fused=False):
        import sys
        import types

        from trnfft import fft_core as fc
        from trnfft.nki import dispatch, four_step

        fake_xla = types.ModuleType("torch_xla")
        fake_xla.device = lambda: torch.device(device)
        monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)
        fake_bfly = types.ModuleType("trnfft.nki.butterfly")
        fake_bfly.butterfly_stage_kernel = None
        fake_bfly.butterfly_stage_kernel_kahan = None
        monkeypatch.setitem(sys.modules, "trnfft.nki.butterfly", fake_bfly)

        calls = {"gemm": 0, "gemm_tn": 0, "mul": 0, "four_step": 0}

        def fake_gemm_tn(at_re, at_im, b_re, b_im):
            # Mirrors _complex_gemm_tn_kernel: at is (G*K, M), block g = A_g^T.
            calls["gemm_tn"] += 1
            calls.setdefault("tn_rows", []).append(at_re.shape[0])
            k, n = b_re.shape
            gk, m = at_re.shape
            assert gk % k == 0 and (m <= 128 or m % 128 == 0) and (k <= 128 or k % 128 == 0)
            a_re = at_re.reshape(gk // k, k, m).transpose(1, 2).reshape(-1, k)
            a_im = at_im.reshape(gk // k, k, m).transpose(1, 2).reshape(-1, k)
            return a_re @ b_re - a_im @ b_im, a_re @ b_im + a_im @ b_re

        def fake_gemm(a_re, a_im, b_re, b_im):
            calls["gemm"] += 1
            for t in (a_re, a_im, b_re, b_im):
                if t.device != torch.device(device):
                    raise RuntimeError(f"input tensor is on {t.device} device, expected {device}")
            # Mirror _complex_gemm_kernel's M tiling (min(M, 128), no tail tile).
            m = a_re.shape[0]
            assert m <= 128 or m % 128 == 0, f"M={m} not tileable"
            return a_re @ b_re - a_im @ b_im, a_re @ b_im + a_im @ b_re

        def fake_mul(a_re, a_im, b_re, b_im):
            calls["mul"] += 1
            return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re

        def fake_four_step(x_re, x_im, w1_blk, t_re, t_im, w2_re, w2_im):
            # Mirrors four_step_fft_kernel: X[b] as (N1, N2), Y = X^T W1,
            # Y *= T, O = W2^T Y, O[b] as (N2, N1) is the natural-order output.
            calls["four_step"] += 1
            calls.setdefault("rows", []).append(x_re.shape[0])
            b, n = x_re.shape
            n1 = w1_blk.shape[0] // 2
            n2 = n // n1
            p, q = w2_re.shape[0] // n2, t_re.shape[1] // n1
            assert n1 * n2 == n and n1 <= 64 and p * n2 <= 128 and b % p == 0
            assert t_re.shape == (p * n2, q * n1) and q * n1 <= 512
            if device == "meta":
                return torch.empty_like(x_re), torch.empty_like(x_im)
            w1r, w1i = w1_blk[:n1, :n1], w1_blk[:n1, n1:]
            assert torch.equal(w1_blk[n1:, :n1], -w1i) and torch.equal(w1_blk[n1:, n1:], w1r)
            assert torch.equal(t_re, t_re[:n2, :n1].repeat(p, q))
            w2r, w2i = w2_re[:n2, :n2], w2_im[:n2, :n2]
            assert torch.equal(w2_re, torch.block_diag(*[w2r] * p))
            X = torch.complex(x_re, x_im).reshape(b, n1, n2)
            Y = torch.einsum("bij,ik->bjk", X, torch.complex(w1r, w1i))
            Y = Y * torch.complex(t_re[:n2, :n1], t_im[:n2, :n1])
            O = torch.einsum("jl,bjk->blk", torch.complex(w2r, w2i), Y).reshape(b, n)
            return O.real.contiguous(), O.imag.contiguous()

        monkeypatch.setattr(dispatch, "HAS_NKI", True)
        monkeypatch.setattr(dispatch, "_complex_gemm_kernel", fake_gemm, raising=False)
        monkeypatch.setattr(dispatch, "_complex_gemm_tn_kernel", fake_gemm_tn, raising=False)
        monkeypatch.setattr(dispatch, "_complex_mul_kernel", fake_mul, raising=False)
        monkeypatch.setattr(four_step, "four_step_fft_kernel", fake_four_step, raising=False)
        monkeypatch.setattr(fc, "_FOUR_STEP_BATCH_FUSED", batch_fused)
        monkeypatch.setattr(dispatch, "_use_nki", lambda: True)
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        monkeypatch.setattr(fc, "_use_nki", lambda: True)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 256)
        monkeypatch.setattr(fc, "_FOUR_STEP_MAX", 4096)

        def _no_stockham(*a, **k):
            raise AssertionError("Stockham path reached; expected four-step")

        for name in (
            "_fft_via_stockham_nki",
            "_fft_via_stockham_nki_r8",
            "_fft_via_stockham_nki_mixed",
        ):
            monkeypatch.setattr(fc, name, _no_stockham)
        return calls

    @staticmethod
    def _rel(y, ref):
        got = y.real.double().numpy() + 1j * y.imag.double().numpy()
        return np.linalg.norm(got - ref) / np.linalg.norm(ref)

    @pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
    @pytest.mark.parametrize("batch", [1, 32, 33])
    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("fused", [False, True])
    def test_matches_numpy_two_gemm_launches(self, monkeypatch, n, batch, inverse, fused):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_FOUR_STEP_FUSED_TWIDDLE", fused)
        g = torch.Generator().manual_seed(n + batch)
        xr, xi = torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g)
        y = (trnfft.ifft if inverse else trnfft.fft)(ComplexTensor(xr, xi))
        xc = xr.double().numpy() + 1j * xi.double().numpy()
        ref = np.fft.ifft(xc) if inverse else np.fft.fft(xc)
        assert y.shape == (batch, n)
        assert self._rel(y, ref) < 1e-5
        assert calls["gemm"] == 2

    def test_fused_twiddle_uses_one_mul_kernel(self, monkeypatch):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_FOUR_STEP_FUSED_TWIDDLE", True)
        xr, xi = torch.randn(32, 1024), torch.randn(32, 1024)
        y = trnfft.fft(ComplexTensor(xr, xi))
        ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
        assert self._rel(y, ref) < 1e-5
        assert calls == {"gemm": 2, "gemm_tn": 0, "mul": 1, "four_step": 0}

    def test_four_step_max_zero_restores_stockham(self, monkeypatch):
        from trnfft import fft_core as fc

        self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_FOUR_STEP_MAX", 0)
        with pytest.raises(AssertionError, match="Stockham path reached"):
            trnfft.fft(ComplexTensor(torch.randn(4, 1024), torch.randn(4, 1024)))

    def test_threshold_zero_disables_four_step(self, monkeypatch):
        from trnfft import fft_core as fc

        self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 0)
        with pytest.raises(AssertionError, match="Stockham path reached"):
            trnfft.fft(ComplexTensor(torch.randn(4, 1024), torch.randn(4, 1024)))

    def test_raised_threshold_single_gemm(self, monkeypatch):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1024)
        xr, xi = torch.randn(32, 1024), torch.randn(32, 1024)
        y = trnfft.ifft(ComplexTensor(xr, xi))
        ref = np.fft.ifft(xr.double().numpy() + 1j * xi.double().numpy())
        assert self._rel(y, ref) < 1e-5
        assert calls["gemm"] == 1

    def test_stays_on_caller_device(self, monkeypatch):
        self._install_fakes(monkeypatch, device="meta")
        x = ComplexTensor(
            torch.empty(32, 1024, device="meta"), torch.empty(32, 1024, device="meta")
        )
        y = trnfft.fft(x)
        assert y.real.device.type == "meta" and y.imag.device.type == "meta"
        assert y.shape == (32, 1024)

    def test_env_knobs_parse(self, monkeypatch):
        from trnfft import fft_core as fc

        monkeypatch.setenv("TRNFFT_TEST_KNOB", "1024")
        assert fc._env_int("TRNFFT_TEST_KNOB", 256) == 1024
        monkeypatch.setenv("TRNFFT_TEST_KNOB", "")
        assert fc._env_int("TRNFFT_TEST_KNOB", 256) == 256
        monkeypatch.setenv("TRNFFT_TEST_KNOB", "abc")
        with pytest.warns(UserWarning):
            assert fc._env_int("TRNFFT_TEST_KNOB", 256) == 256

    def test_defaults(self):
        """Defaults chosen from the Trn2 A/B (trnfft#82)."""
        import os

        from trnfft import fft_core as fc

        knobs = (
            "TRNFFT_DFT_GEMM_THRESHOLD",
            "TRNFFT_FOUR_STEP_MAX",
            "TRNFFT_FOUR_STEP_FUSED_TWIDDLE",
            "TRNFFT_FOUR_STEP_BATCH_FUSED",
            "TRNFFT_FOUR_STEP_FUSED_MIN",
            "TRNFFT_FOUR_STEP_FUSED_MAX_ROWS",
        )
        if any(os.environ.get(k) for k in knobs):
            pytest.skip("trnfft env knob set; defaults overridden")

        assert fc._env_int("TRNFFT_DFT_GEMM_THRESHOLD", 1024) == fc._DFT_GEMM_THRESHOLD
        assert fc._DFT_GEMM_THRESHOLD == 1024
        assert fc._FOUR_STEP_MAX == 4096
        assert fc._FOUR_STEP_FUSED_TWIDDLE is True
        assert fc._FOUR_STEP_BATCH_FUSED is True
        assert fc._FOUR_STEP_FUSED_MIN == 512
        assert fc._FOUR_STEP_FUSED_MAX_ROWS == 4096
        assert fc._LOWP_GEMM_MAX == 256

    @pytest.mark.parametrize("n,expect_gemms", [(1024, 1), (2048, 2), (4096, 2)])
    def test_default_threshold_routing(self, monkeypatch, n, expect_gemms):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1024)
        xr, xi = torch.randn(32, n), torch.randn(32, n)
        y = trnfft.fft(ComplexTensor(xr, xi))
        ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
        assert self._rel(y, ref) < 1e-5
        assert calls["gemm"] == expect_gemms

    def test_bf16_mode_keeps_256_cap(self, monkeypatch):
        """bf16 above N=256 must not take the BF16 GEMM even with threshold 1024."""
        from trnfft import fft_core as fc

        self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1024)

        def _no_bf16(*a, **k):
            raise AssertionError("bf16 GEMM reached above _LOWP_GEMM_MAX")

        monkeypatch.setattr(fc, "_fft_via_gemm_bf16", _no_bf16)
        x = ComplexTensor(torch.randn(4, 512), torch.randn(4, 512))
        fc._cooley_tukey_nki_nograd(x, inverse=False, precision="bf16")

    @pytest.mark.parametrize("n,expect_gemms", [(1024, 1), (4096, 2)])
    def test_kahan_uses_gemm_paths(self, monkeypatch, n, expect_gemms):
        """kahan must not be less accurate than fast: it shares the GEMM paths (#82)."""
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1024)
        xr, xi = torch.randn(4, n), torch.randn(4, n)
        y = fc._cooley_tukey_nki_nograd(ComplexTensor(xr, xi), inverse=False, precision="kahan")
        ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
        assert self._rel(y, ref) < 1e-5
        assert calls["gemm"] == expect_gemms

    @pytest.mark.parametrize("n", [997, 1009])
    def test_kahan_bluestein_not_worse_than_fast(self, monkeypatch, n):
        """CPU mirror of the neuron TestKahanCharacterization contract."""
        from trnfft import fft_core as fc
        from trnfft import get_precision, set_precision

        self._install_fakes(monkeypatch)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1024)
        torch.manual_seed(42)
        x = torch.randn(n)
        ref = np.fft.fft(x.numpy().astype(np.float64))
        errs = {}
        old = get_precision()
        try:
            for prec in ("fast", "kahan"):
                set_precision(prec)
                y = trnfft.fft(x)
                got = y.real.double().numpy() + 1j * y.imag.double().numpy()
                errs[prec] = np.abs(got - ref).max() / np.abs(ref).max()
        finally:
            set_precision(old)
        assert errs["kahan"] <= 1.5 * errs["fast"], errs


class TestFourStepBatchFused(TestFourStepDispatch):
    """Batch-fused four-step: one ``four_step_fft_kernel`` launch per call.

    Reuses the TestFourStepDispatch fakes with a stand-in for the fused kernel
    that checks its operand layout and mirrors its math.
    """

    @pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
    @pytest.mark.parametrize("batch", [1, 9, 33])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_matches_numpy_one_launch(self, monkeypatch, n, batch, inverse):
        calls = self._install_fakes(monkeypatch, batch_fused=True)
        g = torch.Generator().manual_seed(n + batch)
        xr, xi = torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g)
        y = (trnfft.ifft if inverse else trnfft.fft)(ComplexTensor(xr, xi))
        xc = xr.double().numpy() + 1j * xi.double().numpy()
        ref = np.fft.ifft(xc) if inverse else np.fft.fft(xc)
        assert y.shape == (batch, n)
        assert self._rel(y, ref) < 1e-5
        assert (calls["gemm"], calls["mul"], calls["four_step"]) == (0, 0, 1)

    def test_leading_dims_preserved(self, monkeypatch):
        calls = self._install_fakes(monkeypatch, batch_fused=True)
        xr, xi = torch.randn(3, 5, 2048), torch.randn(3, 5, 2048)
        y = trnfft.fft(ComplexTensor(xr, xi))
        ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
        assert y.shape == (3, 5, 2048)
        assert self._rel(y, ref) < 1e-5
        assert calls["four_step"] == 1

    @pytest.mark.parametrize(
        "n,expect", [(256, (1, 0)), (512, (0, 1)), (1024, (0, 1)), (2048, (0, 1)), (4096, (0, 1))]
    )
    def test_default_threshold_routing_fused(self, monkeypatch, n, expect):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch, batch_fused=True)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1024)
        trnfft.fft(ComplexTensor(torch.randn(16, n), torch.randn(16, n)))
        assert (calls["gemm"], calls["four_step"]) == expect

    def test_fused_matches_unfused(self, monkeypatch):
        """Same constants and math: the two drivers agree to FP32 rounding."""
        from trnfft import fft_core as fc

        self._install_fakes(monkeypatch, batch_fused=True)
        xr, xi = torch.randn(17, 4096), torch.randn(17, 4096)
        x = ComplexTensor(xr, xi)
        a = fc._fft_via_four_step_fused(x, inverse=False)
        b = fc._fft_via_four_step(x, inverse=False)
        scale = b.real.abs().max()
        assert (a.real - b.real).abs().max() / scale < 1e-5
        assert (a.imag - b.imag).abs().max() / scale < 1e-5

    def test_non_fp32_uses_unfused(self, monkeypatch):
        calls = self._install_fakes(monkeypatch, batch_fused=True)
        x = ComplexTensor(torch.randn(4, 2048).double(), torch.randn(4, 2048).double())
        from trnfft import fft_core as fc

        fc._cooley_tukey_nki_nograd(x, inverse=False)
        assert calls["four_step"] == 0 and calls["gemm"] == 2

    def test_knob_off_uses_unfused(self, monkeypatch):
        calls = self._install_fakes(monkeypatch, batch_fused=False)
        trnfft.fft(ComplexTensor(torch.randn(4, 4096), torch.randn(4, 4096)))
        assert calls["four_step"] == 0 and calls["gemm"] == 2

    def test_stays_on_caller_device_fused(self, monkeypatch):
        calls = self._install_fakes(monkeypatch, device="meta", batch_fused=True)
        x = ComplexTensor(
            torch.empty(32, 4096, device="meta"), torch.empty(32, 4096, device="meta")
        )
        y = trnfft.fft(x)
        assert y.real.device.type == "meta" and y.imag.device.type == "meta"
        assert y.shape == (32, 4096)
        assert calls["four_step"] == 1

    @pytest.mark.parametrize(
        "n2,batch,expect", [(64, 1, 1), (64, 1024, 2), (64, 33, 1), (32, 6, 2), (16, 24, 8)]
    )
    def test_pack_rows(self, n2, batch, expect):
        from trnfft.nki.four_step import four_step_pack_rows

        assert four_step_pack_rows(n2, batch) == expect

    @pytest.mark.parametrize(
        "n1,batch,pack,expect", [(64, 2, 2, 1), (64, 1024, 2, 8), (32, 64, 4, 16)]
    )
    def test_group_packs(self, n1, batch, pack, expect):
        from trnfft.nki.four_step import four_step_group_packs

        assert four_step_group_packs(n1, batch, pack) == expect

    def test_fits(self):
        from trnfft import fft_core as fc

        assert fc._four_step_fused_fits(4096)
        assert not fc._four_step_fused_fits(8192)

    def test_fused_min_knob(self, monkeypatch):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch, batch_fused=True)
        monkeypatch.setattr(fc, "_DFT_GEMM_THRESHOLD", 1024)
        monkeypatch.setattr(fc, "_FOUR_STEP_FUSED_MIN", 2048)
        trnfft.fft(ComplexTensor(torch.randn(4, 1024), torch.randn(4, 1024)))
        assert (calls["gemm"], calls["four_step"]) == (1, 0)

    def test_row_chunks(self, monkeypatch):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch, batch_fused=True)
        monkeypatch.setattr(fc, "_FOUR_STEP_FUSED_MAX_ROWS", 8)
        xr, xi = torch.randn(21, 2048), torch.randn(21, 2048)
        y = trnfft.fft(ComplexTensor(xr, xi))
        ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
        assert self._rel(y, ref) < 1e-5
        assert calls["rows"] == [8, 8, 5]


class TestSixStep(TestFourStepDispatch):
    """Six-step large-N FFT (trnfft#54; default from N=8192 since #94).

    Each row pass of length <= _FOUR_STEP_MAX goes to the batch-fused
    four-step stand-in (or the single GEMM at <= 256); Stockham must not run.
    """

    @pytest.mark.parametrize("n", [8192, 16384, 65536])
    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_matches_numpy(self, monkeypatch, n, batch, inverse):
        calls = self._install_fakes(monkeypatch, batch_fused=True)
        monkeypatch.setenv("TRNFFT_SIX_STEP", "1")
        g = torch.Generator().manual_seed(n + batch)
        xr, xi = torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g)
        y = (trnfft.ifft if inverse else trnfft.fft)(ComplexTensor(xr, xi))
        xc = xr.double().numpy() + 1j * xi.double().numpy()
        ref = np.fft.ifft(xc) if inverse else np.fft.fft(xc)
        assert y.shape == (batch, n)
        assert self._rel(y, ref) < 1e-5
        # Two row passes; at N = 8192 the 64-point pass is a single GEMM, read
        # straight from x's (B, N2, N1) view (trnfft#97).
        assert calls["four_step"] + calls["gemm"] + calls["gemm_tn"] == 2
        assert calls["gemm"] == 0 and calls["gemm_tn"] == (1 if n == 8192 else 0)

    def test_recurses_above_four_step_max(self, monkeypatch):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch, batch_fused=True)
        monkeypatch.setattr(fc, "_FOUR_STEP_MAX", 512)
        monkeypatch.setenv("TRNFFT_SIX_STEP", "1")
        # 2^20 > 512^2: 512 (fused) x 2048, and 2048 recurses as 64 x 32 GEMMs.
        xr, xi = torch.randn(1, 1 << 20), torch.randn(1, 1 << 20)
        y = trnfft.fft(ComplexTensor(xr, xi))
        ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
        assert self._rel(y, ref) < 1e-5
        # The 2048-point pass's own first pass (32-point rows) reads x's view.
        assert (calls["four_step"], calls["gemm_tn"], calls["gemm"]) == (1, 1, 1)

    def test_gemm_pass_row_chunks(self, monkeypatch):
        from trnfft import fft_core as fc

        calls = self._install_fakes(monkeypatch, batch_fused=True)
        monkeypatch.setattr(fc, "_SIX_STEP_GEMM_MAX_ROWS", 128)
        monkeypatch.setenv("TRNFFT_SIX_STEP", "1")
        xr, xi = torch.randn(3, 8192), torch.randn(3, 8192)
        y = trnfft.fft(ComplexTensor(xr, xi))
        ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
        assert self._rel(y, ref) < 1e-5
        # 8192 = 128 x 64: 384 rows of 64 on the GEMM (3 launches of 128 rows,
        # one batch row of x each), 192 -> 256 rows of 128 on the fused kernel
        # (from _SIX_STEP_FUSED_MIN).
        assert (calls["gemm_tn"], calls["gemm"], calls["four_step"]) == (3, 0, 1)
        assert calls["tn_rows"] == [64] * 3
        # TRNFFT_SIX_STEP_TN=0: the transpose and the load_transpose2d GEMM.
        monkeypatch.setenv("TRNFFT_SIX_STEP_TN", "0")
        y0 = trnfft.fft(ComplexTensor(xr, xi))
        assert (calls["gemm_tn"], calls["gemm"], calls["four_step"]) == (3, 3, 2)
        torch.testing.assert_close(y0.real, y.real, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(y0.imag, y.imag, rtol=1e-5, atol=1e-4)

    @pytest.mark.parametrize("inverse", [False, True])
    def test_pass1_tn_matches_transposed_pass(self, monkeypatch, inverse):
        from trnfft import fft_core as fc

        self._install_fakes(monkeypatch, batch_fused=True)
        g = torch.Generator().manual_seed(97)
        B, n1, n2 = 5, 128, 64
        re, im = torch.randn(B, n2 * n1, generator=g), torch.randn(B, n2 * n1, generator=g)
        got = fc._six_step_pass1_tn(re, im, B, n1, n2, inverse)
        a_re = re.reshape(B, n2, n1).transpose(1, 2).reshape(B * n1, n2)
        a_im = im.reshape(B, n2, n1).transpose(1, 2).reshape(B * n1, n2)
        xc = a_re.double().numpy() + 1j * a_im.double().numpy()
        ref = np.fft.ifft(xc) if inverse else np.fft.fft(xc)
        np.testing.assert_allclose(got[0].double().numpy(), ref.real, atol=2e-5)
        np.testing.assert_allclose(got[1].double().numpy(), ref.imag, atol=2e-5)

    def test_pass1_tn_gate(self, monkeypatch):
        from trnfft import fft_core as fc

        self._install_fakes(monkeypatch, batch_fused=True)
        assert fc._six_step_pass1_tn_ok(128, 64)  # N = 2^13
        assert not fc._six_step_pass1_tn_ok(128, 128)  # fused range
        assert not fc._six_step_pass1_tn_ok(192, 64)  # M not tileable
        monkeypatch.setenv("TRNFFT_SIX_STEP_TN", "0")
        assert not fc._six_step_pass1_tn_ok(128, 64)

    def test_fused_rows_per_launch(self):
        from trnfft import fft_core as fc

        cap = fc._FOUR_STEP_FUSED_MAX_ROWS
        assert fc._six_step_fused_rows(4096) == cap
        assert fc._six_step_fused_rows(1024) == 4 * cap
        assert fc._six_step_fused_rows(128) == 32 * cap

    def test_default_above_threshold(self, monkeypatch):
        from trnfft import fft_core as fc

        monkeypatch.delenv("TRNFFT_SIX_STEP", raising=False)
        x = ComplexTensor(torch.randn(1, 8192), torch.randn(1, 8192))
        assert fc._SIX_STEP_MIN == 8192
        assert fc._use_six_step(x, 8192)
        assert not fc._use_six_step(x, 4096)  # fused four-step range
        assert not fc._use_six_step(x, 8192, "double")
        x64 = ComplexTensor(torch.randn(1, 8192).double(), torch.randn(1, 8192).double())
        assert not fc._use_six_step(x64, 8192)
        monkeypatch.setattr(fc, "_SIX_STEP_MIN", 1 << 20)
        assert not fc._use_six_step(x, 8192)
        monkeypatch.setenv("TRNFFT_SIX_STEP", "1")  # forces it below the threshold
        assert fc._use_six_step(x, 8192)
        monkeypatch.setenv("TRNFFT_SIX_STEP", "0")
        assert not fc._use_six_step(x, 1 << 21)

    def test_default_routes_to_six_step(self, monkeypatch):
        calls = self._install_fakes(monkeypatch, batch_fused=True)
        monkeypatch.delenv("TRNFFT_SIX_STEP", raising=False)
        y = trnfft.fft(ComplexTensor(torch.randn(8192), torch.randn(8192)))
        assert y.shape == (8192,)
        assert calls["four_step"] + calls["gemm_tn"] == 2

    def test_env_zero_routes_to_stockham(self, monkeypatch):
        self._install_fakes(monkeypatch, batch_fused=True)
        monkeypatch.setenv("TRNFFT_SIX_STEP", "0")
        with pytest.raises(AssertionError, match="Stockham path reached"):
            trnfft.fft(ComplexTensor(torch.randn(8192), torch.randn(8192)))


class TestBluesteinChirpPrecision:
    """Chirp built in FP64 with k^2 mod 2N (trnfft#82 hardware finding).

    The FP32 chirp angle pi*k^2/N gave 1.2e-4 rel error at N=1000 (Trn2 and
    CPU) and 4.4e-4 at N=4097; with the reduction CPU gives ~4e-7.
    """

    @pytest.mark.parametrize("n", [7, 1000, 4097])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_matches_numpy(self, n, inverse):
        g = torch.Generator().manual_seed(n)
        x = torch.randn(4, n, generator=g)
        y = trnfft.ifft(x) if inverse else trnfft.fft(x)
        ref = (np.fft.ifft if inverse else np.fft.fft)(x.double().numpy())
        got = y.real.double().numpy() + 1j * y.imag.double().numpy()
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert rel < 2e-6, f"N={n}: rel err {rel:.2e}"


class TestStockhamRowChunking:
    """Row-chunked Stockham kernel launches (trnfft#83).

    N >= 2^25 overflowed the NKI kernel body on Trn2 ("body blob exceeds
    uint32 offset range") because a stage launch carried 2^23 rows. Dispatch
    now caps rows per launch (TRNFFT_MAX_ROWS_PER_LAUNCH). Stand-in kernels
    mirror the stage kernels row for row in torch, so chunked and unchunked
    runs must agree bit for bit.
    """

    def test_planner_single_chunk_when_under_cap(self):
        from trnfft.fft_core import _plan_row_chunks

        assert _plan_row_chunks(64, 128) == [(0, 64)]
        assert _plan_row_chunks(1 << 21, 1 << 21) == [(0, 1 << 21)]

    def test_planner_splits_on_tile_boundaries(self):
        from trnfft.fft_core import _plan_row_chunks

        assert _plan_row_chunks(1 << 23, 1 << 21) == [(i << 21, (i + 1) << 21) for i in range(4)]
        # Non-power-of-2 batch padding (B_pad = 384): tail chunk, all on 128s.
        chunks = _plan_row_chunks(384 * 64, 1000)
        assert chunks[0] == (0, 896)
        assert chunks[-1][1] == 384 * 64
        assert all(a % 128 == 0 and b - a <= 896 for a, b in chunks)
        assert all(c[1] == d[0] for c, d in zip(chunks, chunks[1:], strict=False))
        # Cap rounds down to a multiple of 128, minimum 128.
        assert _plan_row_chunks(512, 1) == [(0, 128), (128, 256), (256, 384), (384, 512)]
        with pytest.raises(ValueError):
            _plan_row_chunks(0, 128)

    def test_cap_env_var(self, monkeypatch):
        from trnfft import fft_core as fc

        monkeypatch.delenv("TRNFFT_MAX_ROWS_PER_LAUNCH", raising=False)
        assert fc._max_rows_per_launch() == 1 << 21
        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", "300")
        assert fc._max_rows_per_launch() == 256
        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", "5")
        assert fc._max_rows_per_launch() == 128

    def _install_fakes(self, monkeypatch):
        import sys
        import types

        from trnfft import fft_core as fc

        fake_xla = types.ModuleType("torch_xla")
        fake_xla.device = lambda: torch.device("cpu")
        monkeypatch.setitem(sys.modules, "torch_xla", fake_xla)

        launches = []

        def _check(t):
            g = t.shape[0]
            assert g <= 128 or g % 128 == 0, f"rows={g} not tileable"

        # Elementwise per-row arithmetic (no BLAS, whose blocking can depend
        # on the row count), so any chunking difference would show up.
        def _rowwise_dft(a_re, a_im, w_re, w_im):
            r = a_re.shape[1]
            y_re = torch.zeros_like(a_re)
            y_im = torch.zeros_like(a_im)
            for k in range(r):
                ar, ai = a_re[:, k : k + 1], a_im[:, k : k + 1]
                y_re = y_re + (ar * w_re[k] - ai * w_im[k])
                y_im = y_im + (ar * w_im[k] + ai * w_re[k])
            return y_re, y_im

        def _w(r):
            k = torch.arange(r, dtype=torch.float64)
            ang = -2.0 * np.pi * torch.outer(k, k) / r
            return torch.cos(ang).float(), torch.sin(ang).float()

        w4_re, w4_im = _w(4)

        def fake_r4(x_re, x_im, tw_re, tw_im):
            _check(x_re)
            launches.append(("r4", x_re.shape[0]))
            a_re = x_re * tw_re - x_im * tw_im
            a_im = x_re * tw_im + x_im * tw_re
            return _rowwise_dft(a_re, a_im, w4_re, w4_im)

        def fake_r8(a_re, a_im, w8_re, w8_im):
            _check(a_re)
            assert w8_re.shape == (8, 8)
            launches.append(("r8", a_re.shape[0]))
            return _rowwise_dft(a_re, a_im, w8_re, w8_im)

        fake_st = types.ModuleType("trnfft.nki.stockham")
        fake_st.stockham_radix4_stage_kernel = fake_r4
        fake_st.stockham_radix8_w8_kernel = fake_r8
        monkeypatch.setitem(sys.modules, "trnfft.nki.stockham", fake_st)
        from trnfft.nki import dispatch

        monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
        return fc, launches

    @pytest.mark.parametrize(
        "driver,n",
        [
            ("_fft_via_stockham_nki_mixed", 2048),
            ("_fft_via_stockham_nki_mixed", 8192),
            ("_fft_via_stockham_nki", 1024),
            ("_fft_via_stockham_nki_r8", 4096),
        ],
    )
    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_chunked_bit_identical_to_unchunked(self, monkeypatch, driver, n, batch, inverse):
        fc, launches = self._install_fakes(monkeypatch)
        fn = getattr(fc, driver)
        g = torch.Generator().manual_seed(n + batch)
        x = ComplexTensor(torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g))

        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", str(1 << 30))
        whole = fn(x, inverse)
        n_whole = len(launches)
        launches.clear()

        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", "128")
        chunked = fn(x, inverse)
        assert len(launches) > n_whole
        assert max(rows for _, rows in launches) <= 128

        assert torch.equal(chunked.real, whole.real)
        assert torch.equal(chunked.imag, whole.imag)
        xc = x.real.double().numpy() + 1j * x.imag.double().numpy()
        ref = np.fft.ifft(xc) if inverse else np.fft.fft(xc)
        got = chunked.real.double().numpy() + 1j * chunked.imag.double().numpy()
        assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-5


@pytest.mark.neuron
class TestLargeFFTNeuron:
    """N = 2^25 on device (trnfft#83): stage launches are row-chunked."""

    @pytest.mark.parametrize("six_step", [None, "0"])
    def test_fft_2_25_vs_torch(self, nki_backend, monkeypatch, six_step):
        """Default dispatch (six-step since #94) and ``TRNFFT_SIX_STEP=0`` (Stockham)."""
        if six_step is None:
            monkeypatch.delenv("TRNFFT_SIX_STEP", raising=False)
        else:
            monkeypatch.setenv("TRNFFT_SIX_STEP", six_step)
        n = 1 << 25
        g = torch.Generator().manual_seed(83)
        x = torch.randn(n, generator=g)
        y = trnfft.fft(x)
        ref = torch.fft.fft(x.double())
        got = torch.complex(y.real.double().cpu(), y.imag.double().cpu())
        rel = (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()
        assert rel < 1e-5, f"N=2^25 rel err {rel:.2e}"

    def test_chunked_bit_identical_2_22(self, nki_backend, monkeypatch):
        """N = 2^22 fits one launch per stage (max 2^20 rows); forcing the cap
        to TRNFFT_BITID_CAP rows (default 2^17) must not change a single bit."""
        import os

        n = 1 << 22
        cap = os.environ.get("TRNFFT_BITID_CAP", str(1 << 17))
        monkeypatch.setenv("TRNFFT_SIX_STEP", "0")  # Stockham row chunking under test
        g = torch.Generator().manual_seed(84)
        x = torch.randn(n, generator=g)
        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", str(1 << 30))
        whole = trnfft.fft(x)
        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", cap)
        chunked = trnfft.fft(x)
        assert torch.equal(chunked.real.cpu(), whole.real.cpu())
        assert torch.equal(chunked.imag.cpu(), whole.imag.cpu())


@pytest.mark.neuron
class TestFourStepFusedHostNeuron:
    """Host input on the fused four-step path moves to the device once (trnfft#94)."""

    @pytest.mark.parametrize("inverse", [False, True])
    def test_host_matches_device_bitwise(self, nki_backend, inverse):
        # 9000 rows: three launches of <= 4096 rows, the last one partial.
        g = torch.Generator().manual_seed(94)
        xr, xi = torch.randn(9000, 2048, generator=g), torch.randn(9000, 2048, generator=g)
        f = trnfft.ifft if inverse else trnfft.fft
        host = f(ComplexTensor(xr, xi))
        assert host.real.device.type == "cpu"
        dev = torch.device("neuron")
        on_dev = f(ComplexTensor(xr.to(dev), xi.to(dev)))
        assert on_dev.real.device.type == "neuron"
        assert torch.equal(host.real, on_dev.real.cpu())
        assert torch.equal(host.imag, on_dev.imag.cpu())


@pytest.mark.neuron
class TestSixStepNeuron:
    """Six-step large-N FFT on device (trnfft#54; ``TRNFFT_SIX_STEP=1`` forces it,
    and it is the default from N=8192 since #94)."""

    @staticmethod
    def _rel(y, ref):
        got = torch.complex(y.real.double().cpu(), y.imag.double().cpu())
        return (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()

    @pytest.mark.parametrize("batch,log2n", [(1, 13), (1, 22), (4, 16), (1, 25)])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_vs_torch_fp64(self, nki_backend, monkeypatch, batch, log2n, inverse):
        monkeypatch.setenv("TRNFFT_SIX_STEP", "1")
        g = torch.Generator().manual_seed(54 + log2n)
        xr = torch.randn(batch, 1 << log2n, generator=g)
        xi = torch.randn(batch, 1 << log2n, generator=g)
        dev = torch.device("neuron")
        y = (trnfft.ifft if inverse else trnfft.fft)(ComplexTensor(xr.to(dev), xi.to(dev)))
        assert y.real.device.type == "neuron"
        x64 = torch.complex(xr.double(), xi.double())
        ref = torch.fft.ifft(x64) if inverse else torch.fft.fft(x64)
        assert self._rel(y, ref) < 1e-6

    @pytest.mark.parametrize("inverse", [False, True])
    def test_default_dispatch_is_six_step(self, nki_backend, monkeypatch, inverse):
        """With TRNFFT_SIX_STEP unset, N=2^20 takes the six-step (#94). Its error
        (~2e-7) is ~10x below the Stockham path's (~3e-6), so 1e-6 tells them apart."""
        monkeypatch.delenv("TRNFFT_SIX_STEP", raising=False)
        g = torch.Generator().manual_seed(9420)
        xr, xi = torch.randn(1 << 20, generator=g), torch.randn(1 << 20, generator=g)
        dev = torch.device("neuron")
        y = (trnfft.ifft if inverse else trnfft.fft)(ComplexTensor(xr.to(dev), xi.to(dev)))
        x64 = torch.complex(xr.double(), xi.double())
        ref = torch.fft.ifft(x64) if inverse else torch.fft.fft(x64)
        assert self._rel(y, ref) < 1e-6

    def test_default_rfft_irfft_fft2_bluestein(self, nki_backend, monkeypatch):
        """rfft / irfft, fft2 and Bluestein reach the six-step through fft_core."""
        monkeypatch.delenv("TRNFFT_SIX_STEP", raising=False)
        g = torch.Generator().manual_seed(9421)
        x = torch.randn(1 << 18, generator=g)
        X = trnfft.rfft(x)
        ref = torch.fft.rfft(x.double())
        assert self._rel(X, ref) < 1e-6
        back = trnfft.irfft(X, n=x.shape[-1])
        assert (torch.linalg.vector_norm(back.double().cpu() - x.double()) / x.norm()).item() < 1e-6
        a = torch.randn(8, 1 << 14, generator=g)
        assert self._rel(trnfft.fft2(a), torch.fft.fft2(a.double())) < 1e-6
        b = torch.randn(5000, generator=g)  # Bluestein, padded to 16384
        assert self._rel(trnfft.fft(b), torch.fft.fft(b.double())) < 1e-5

    def test_host_input_returns_to_host(self, nki_backend, monkeypatch):
        monkeypatch.setenv("TRNFFT_SIX_STEP", "1")
        x = torch.randn(1 << 20)
        y = trnfft.fft(x)
        assert y.real.device.type == "cpu"
        assert self._rel(y, torch.fft.fft(x.double())) < 1e-6


@pytest.mark.neuron
class TestMulticoreSplitNeuron:
    """Host batch split across logical NeuronCores (trnfft#54)."""

    def test_host_split_matches_single_core(self, nki_backend):
        import torch_neuronx

        from trnfft.nki.multicore import _batch_split_fft, _six_step_multicore

        if torch_neuronx.device_count() < 2:
            pytest.skip("needs >= 2 visible NeuronCores (NEURON_RT_VISIBLE_CORES)")
        g = torch.Generator().manual_seed(93)
        x = ComplexTensor(torch.randn(256, 1024, generator=g), torch.randn(256, 1024, generator=g))
        one = trnfft.fft(x)
        for _ in range(2):  # first call runs the shards in turn, then concurrently
            y = _batch_split_fft(x, inverse=False, num_cores=2)
            assert y.real.device.type == "cpu"
            torch.testing.assert_close(y.real, one.real.cpu())
            torch.testing.assert_close(y.imag, one.imag.cpu())
        s = torch.randn(1 << 20, generator=g)
        ref = torch.fft.fft(s.double())
        for _ in range(2):
            y = _six_step_multicore(ComplexTensor(s, torch.zeros_like(s)), False, 2)
            got = torch.complex(y.real.double(), y.imag.double())
            rel = (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()
            assert rel < 1e-6


def _same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


@pytest.mark.neuron
class TestPinnedOutputNeuron:
    """Large host results come from the caching host allocator (trnfft#94
    follow-up, as trnblas#64): the same bits as pageable memory."""

    @pytest.mark.parametrize(
        "batch,n",
        [(256, 4096), (64, 1 << 14), (4096, 256), (300, 1 << 13), (1, 1 << 20)],
        ids=["fused-4096", "six-step-2^14", "gemm-256", "six-step-2^13", "six-step-2^20"],
    )
    @pytest.mark.parametrize("inverse", [False, True])
    def test_pinned_and_bit_identical(self, nki_backend, monkeypatch, batch, n, inverse):
        g = torch.Generator().manual_seed(n + batch)
        x = ComplexTensor(torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g))
        f = trnfft.ifft if inverse else trnfft.fft
        monkeypatch.setenv("TRNFFT_PINNED_OUTPUT", "0")
        off = f(x)
        monkeypatch.delenv("TRNFFT_PINNED_OUTPUT")
        on = f(x)
        assert on.real.device.type == "cpu" and on.real.is_pinned() and on.imag.is_pinned()
        assert not off.real.is_pinned()
        assert _same_bits(on.real, off.real) and _same_bits(on.imag, off.imag)

    def test_small_results_stay_pageable(self, nki_backend):
        x = ComplexTensor(torch.randn(8, 4096), torch.randn(8, 4096))  # 128 KiB per plane
        assert not trnfft.fft(x).real.is_pinned()


@pytest.mark.neuron
class TestSixStepPass1TNNeuron:
    """N = 2^13 pass 1 reads x's (B, N2, N1) view on _complex_gemm_tn_kernel
    (trnfft#97): bit-identical to the transpose + load_transpose2d GEMM."""

    @pytest.mark.parametrize("batch", [3, 300])  # 300: two launches (256 + 44 rows of x)
    @pytest.mark.parametrize("inverse", [False, True])
    def test_bit_identical_to_transposed_pass(self, nki_backend, monkeypatch, batch, inverse):
        g = torch.Generator().manual_seed(97 + batch)
        xr, xi = torch.randn(batch, 8192, generator=g), torch.randn(batch, 8192, generator=g)
        dev = torch.device("neuron")
        x = ComplexTensor(xr.to(dev), xi.to(dev))
        f = trnfft.ifft if inverse else trnfft.fft
        monkeypatch.setenv("TRNFFT_SIX_STEP_TN", "0")
        old = f(x)
        monkeypatch.delenv("TRNFFT_SIX_STEP_TN")
        new = f(x)
        assert _same_bits(new.real.cpu(), old.real.cpu())
        assert _same_bits(new.imag.cpu(), old.imag.cpu())


@pytest.mark.neuron
class TestPETransposeNeuron:
    """trnfft#97: the plain-load kernels (``TRNFFT_PE_TRANSPOSE`` default) give
    the same bits as the ``load_transpose2d`` kernels (``=0``) on device."""

    @staticmethod
    def _bits(t):
        return t.contiguous().cpu().view(torch.int32)

    def _both(self, monkeypatch, f, x):
        monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "0")
        old = f(x)
        monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "1")
        new = f(x)
        return old, new

    @pytest.mark.parametrize("batch,n", [(1, 8), (16, 64), (1024, 8), (8192, 32), (512, 128)])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_dft_gemm_bitwise(self, nki_backend, monkeypatch, batch, n, inverse):
        g = torch.Generator().manual_seed(97 * n + batch)
        dev = torch.device("neuron")
        xr, xi = torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g)
        x = ComplexTensor(xr.to(dev), xi.to(dev))
        f = trnfft.ifft if inverse else trnfft.fft
        old, new = self._both(monkeypatch, f, x)
        assert torch.equal(self._bits(old.real), self._bits(new.real))
        assert torch.equal(self._bits(old.imag), self._bits(new.imag))
        ref = torch.fft.ifft if inverse else torch.fft.fft
        want = ref(torch.complex(xr, xi).to(torch.complex128))
        got = torch.complex(new.real.double().cpu(), new.imag.double().cpu())
        assert (torch.linalg.vector_norm(got - want) / torch.linalg.vector_norm(want)) < 1e-6

    @pytest.mark.parametrize("batch,n", [(1000, 64), (200, 8), (130, 256)])
    def test_dft_gemm_ragged_batch(self, nki_backend, batch, n):
        """Batches over 128 rows that are not a multiple of 128 (every row was
        wrong past the last full 128-row tile before trnfft#97)."""
        g = torch.Generator().manual_seed(batch + n)
        xr, xi = torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g)
        y = trnfft.fft(ComplexTensor(xr, xi))
        want = torch.fft.fft(torch.complex(xr, xi).to(torch.complex128))
        got = torch.complex(y.real.double().cpu(), y.imag.double().cpu())
        err = (got - want).abs().amax(dim=1) / want.abs().amax(dim=1)
        assert err.max() < 1e-5, (err > 1e-5).nonzero().flatten()[:4]

    @pytest.mark.parametrize("batch", [1, 64])
    def test_stockham_radix8_bitwise(self, nki_backend, monkeypatch, batch):
        from trnfft import fft_core as fc

        g = torch.Generator().manual_seed(98 + batch)
        x = ComplexTensor(
            torch.randn(batch, 512, generator=g), torch.randn(batch, 512, generator=g)
        )
        old, new = self._both(monkeypatch, lambda v: fc._fft_via_stockham_nki_r8(v, False), x)
        assert torch.equal(self._bits(old.real), self._bits(new.real))
        assert torch.equal(self._bits(old.imag), self._bits(new.imag))


class TestPETransposeSelection:
    """trnfft#97 knobs: ``TRNFFT_PE_TRANSPOSE`` picks the GEMM kernel and
    ``_dft_gemm_pack`` the row packing of the small-N DFT-GEMM (CPU only)."""

    def test_env_forces_kernel(self, monkeypatch):
        from trnfft.nki import dispatch

        monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "0")
        assert dispatch._gemm_kernel_names() == ("_complex_gemm_t2d_kernel", "complex_gemm_t2d")
        monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "1")
        assert dispatch._gemm_kernel_names() == ("_complex_gemm_kernel", "complex_gemm")

    @pytest.mark.parametrize(
        "rows,n,g",
        [(1, 8, 1), (16, 8, 16), (32768, 8, 16), (1024, 64, 2), (8192, 32, 4), (1000, 64, 1)]
        + [(24, 8, 8), (4096, 128, 1), (64, 100, 1), (384, 16, 8), (640, 16, 8), (1280, 16, 2)],
    )
    def test_pack_factor(self, monkeypatch, rows, n, g):
        from trnfft import fft_core as fc

        monkeypatch.setattr(fc, "_use_nki", lambda: True)
        monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "1")
        assert fc._dft_gemm_pack(rows, n) == g
        monkeypatch.setenv("TRNFFT_GEMM_PACK", "0")
        assert fc._dft_gemm_pack(rows, n) == 1
        monkeypatch.delenv("TRNFFT_GEMM_PACK")
        monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "0")
        assert fc._dft_gemm_pack(rows, n) == 1

    @pytest.mark.parametrize("precision", ["bf16", "bf16_refined", "ozaki", "ozaki_hq"])
    @pytest.mark.parametrize(
        "rows,n,shape", [(1000, 8, (1000, 8)), (256, 16, (32, 128)), (1024, 64, (512, 128))]
    )
    def test_packed_bf16_gemm_same_bits(self, monkeypatch, precision, rows, n, shape):
        """The BF16 DFT-GEMMs pack rows for n < 128 (trnfft#101 follow-up).
        With an FP64 stand-in for ``complex_gemm_bf16`` (exact BF16 products,
        like the FP32 PSUM) packed and unpacked give the same bits."""
        from trnfft import fft_core as fc
        from trnfft.nki import dispatch

        seen = []

        def fake_gemm(a, b):
            seen.append((tuple(a.real.shape), tuple(b.real.shape)))
            ar, ai, br, bi = (t.double() for t in (a.real, a.imag, b.real, b.imag))
            return ComplexTensor((ar @ br - ai @ bi).float(), (ar @ bi + ai @ br).float())

        monkeypatch.setattr(fc, "_use_nki", lambda: True)
        monkeypatch.setattr(dispatch, "complex_gemm_bf16", fake_gemm)
        monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "1")
        g = torch.Generator().manual_seed(rows + n)
        x = ComplexTensor(torch.randn(rows, n, generator=g), torch.randn(rows, n, generator=g))
        drivers = {
            "bf16": fc._fft_via_gemm_bf16,
            "bf16_refined": fc._fft_iterative_refinement,
            "ozaki": fc._fft_via_ozaki,
            "ozaki_hq": fc._fft_via_ozaki_hq,
        }
        packed = drivers[precision](x, False)
        assert seen and all(s == (shape, (shape[1], shape[1])) for s in seen)
        monkeypatch.setenv("TRNFFT_GEMM_PACK", "0")
        seen.clear()
        plain = drivers[precision](x, False)
        assert seen and all(s == ((rows, n), (n, n)) for s in seen)
        assert torch.equal(packed.real.view(torch.int32), plain.real.view(torch.int32))
        assert torch.equal(packed.imag.view(torch.int32), plain.imag.view(torch.int32))

    @pytest.mark.parametrize("inverse", [False, True])
    def test_packed_gemm_matches_numpy(self, monkeypatch, inverse):
        """The packed (rows/g, g*n) x block-diagonal W GEMM, run with a CPU
        stand-in for ``complex_gemm``, is the row-wise DFT."""
        from trnfft import fft_core as fc
        from trnfft.nki import dispatch

        seen = []

        def fake_gemm(a, b):
            seen.append((tuple(a.real.shape), tuple(b.real.shape)))
            return ComplexTensor(
                a.real @ b.real - a.imag @ b.imag, a.real @ b.imag + a.imag @ b.real
            )

        monkeypatch.setattr(fc, "_use_nki", lambda: True)
        monkeypatch.setattr(dispatch, "complex_gemm", fake_gemm)
        monkeypatch.setenv("TRNFFT_PE_TRANSPOSE", "1")
        g = torch.Generator().manual_seed(97)
        x = torch.complex(torch.randn(256, 16, generator=g), torch.randn(256, 16, generator=g))
        y = fc._fft_via_gemm(ComplexTensor(x.real.contiguous(), x.imag.contiguous()), inverse)
        assert seen == [((32, 128), (128, 128))]
        ref = (torch.fft.ifft if inverse else torch.fft.fft)(x.to(torch.complex128))
        got = torch.complex(y.real.double(), y.imag.double())
        assert (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)) < 1e-6
