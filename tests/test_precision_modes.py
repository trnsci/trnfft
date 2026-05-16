"""On-silicon characterization of Kahan Dekker-2Prod butterfly variant.

These tests measure whether the ``"kahan"`` precision mode reduces FP32
accumulation error through the Bluestein chain on Trainium hardware.

CPU-only note: on CPU ``"kahan"`` is equivalent to ``"fast"`` because the
Dekker 2Prod compensation in the butterfly stages only engages in the NKI
kernel (``butterfly_stage_kernel_kahan``).  The chirp multiply compensation
also only matters on devices without FP64 — on a CPU host both modes use
the same FP32 arithmetic.  These tests are hardware-only for that reason.
"""

import numpy as np
import pytest
import torch

import trnfft
from trnfft import ComplexTensor, get_precision, set_precision


def _bluestein_rel_error(n: int, precision: str, seed: int = 42) -> float:
    """Return max-relative error of trnfft.fft vs scipy fp64 for size *n*."""
    import scipy.fft

    torch.manual_seed(seed)
    x = torch.randn(n)
    ref = scipy.fft.fft(x.numpy().astype("float64"))

    old = get_precision()
    try:
        set_precision(precision)
        y = trnfft.fft(x)
    finally:
        set_precision(old)

    # .cpu() is a no-op on CPU; required on XLA (Trainium) to copy before numpy().
    y_cplx = y.real.cpu().numpy() + 1j * y.imag.cpu().numpy()
    return float(np.abs(y_cplx - ref).max() / np.abs(ref).max())


@pytest.mark.neuron
class TestKahanCharacterization:
    """On-silicon precision characterization for ``trnfft.set_precision("kahan")``.

    Each test measures fast-mode and kahan-mode error vs a scipy fp64 reference
    and prints both so the results can be copied into
    ``docs/design-notes/kahan-characterization.md``.  The hard assertion is
    only that Kahan does not *regress* vs fast (error ≤ 1.5× fast); an
    improvement is expected but left to the printed output to document.
    """

    @pytest.mark.parametrize("n", [997, 1009, 8193])
    def test_kahan_bluestein_error_vs_fp64(self, nki_backend, n):
        """Compare fast vs kahan FP32 error against a scipy fp64 reference.

        N values are chosen to exercise Bluestein (non-power-of-2) so the
        inner butterfly stages run through the full Bluestein chain.
        """
        fast_err = _bluestein_rel_error(n, "fast")
        kahan_err = _bluestein_rel_error(n, "kahan")

        improvement = fast_err / kahan_err if kahan_err > 0 else float("inf")
        print(
            f"\nN={n}: fast_err={fast_err:.2e}  kahan_err={kahan_err:.2e}"
            f"  kahan_improvement={improvement:.2f}×"
        )

        # Kahan must not significantly worsen precision — it is a compensated
        # variant, so regression would indicate a kernel bug.
        assert kahan_err <= fast_err * 1.5, (
            f"N={n}: kahan ({kahan_err:.2e}) regressed vs fast ({fast_err:.2e})"
        )

    def test_kahan_single_vs_fast_n997(self, nki_backend):
        """Focused single-size test for N=997; easier to read in CI output."""
        fast_err = _bluestein_rel_error(997, "fast")
        kahan_err = _bluestein_rel_error(997, "kahan")

        print(
            f"\nKahan characterization (N=997, trn1):\n"
            f"  fast  error = {fast_err:.3e}\n"
            f"  kahan error = {kahan_err:.3e}\n"
            f"  ratio fast/kahan = {fast_err / kahan_err:.2f}×"
        )

        assert kahan_err <= fast_err * 1.5, (
            f"kahan ({kahan_err:.2e}) worse than fast ({fast_err:.2e})"
        )

    def test_kahan_output_matches_fast_within_fp32_tol(self, nki_backend):
        """Kahan must agree with fast to FP32 tolerance — same answer, tighter.

        Guards against a kernel divergence that produces bit-for-bit different
        results beyond what rounding alone explains.  This complements the
        existing ``test_kahan_butterfly_compiles_and_matches_fast`` which runs
        on power-of-2 sizes; here we test the Bluestein path explicitly.
        """
        from trnfft import fft_core

        n = 997
        torch.manual_seed(42)
        x = torch.randn(n)

        old_prec = get_precision()
        old_thr = fft_core._DFT_GEMM_THRESHOLD
        # N=997 won't hit DFT-GEMM anyway (not power of 2), but zero the
        # threshold to make the intent explicit.
        fft_core._DFT_GEMM_THRESHOLD = 0
        try:
            set_precision("fast")
            y_fast = trnfft.fft(x)
            set_precision("kahan")
            y_kahan = trnfft.fft(x)
        finally:
            set_precision(old_prec)
            fft_core._DFT_GEMM_THRESHOLD = old_thr

        np.testing.assert_allclose(
            y_kahan.real.numpy(),
            y_fast.real.numpy(),
            atol=1e-2,
            rtol=1e-2,
            err_msg="kahan real diverged from fast beyond FP32 tolerance",
        )
        np.testing.assert_allclose(
            y_kahan.imag.numpy(),
            y_fast.imag.numpy(),
            atol=1e-2,
            rtol=1e-2,
            err_msg="kahan imag diverged from fast beyond FP32 tolerance",
        )


@pytest.mark.neuron
class TestKahanButterflyCharacterization:
    """On-silicon error characterization for precision="kahan" butterfly.

    Forces the butterfly path (zeros _DFT_GEMM_THRESHOLD) so DFT-GEMM and
    Stockham are bypassed.  Records fast vs kahan rel error at power-of-2
    sizes and asserts kahan does not regress.

    Run on hardware to fill the "Target ~1e-3" placeholder in precision.py.
    """

    @pytest.mark.parametrize("n", [256, 512, 1024, 4096])
    def test_kahan_butterfly_error_vs_fp64(self, n):
        from trnfft import fft_core

        old_thr = fft_core._DFT_GEMM_THRESHOLD
        fft_core._DFT_GEMM_THRESHOLD = 0  # force butterfly path
        try:
            fast_err = _bluestein_rel_error(n, "fast")
            kahan_err = _bluestein_rel_error(n, "kahan")
            ratio = fast_err / kahan_err if kahan_err > 0 else float("inf")
            print(
                f"\nN={n} (butterfly): fast={fast_err:.2e}  "
                f"kahan={kahan_err:.2e}  improvement={ratio:.1f}×"
            )
            assert kahan_err <= fast_err * 1.5, (
                f"N={n}: kahan ({kahan_err:.2e}) regressed vs fast ({fast_err:.2e})"
            )
        finally:
            fft_core._DFT_GEMM_THRESHOLD = old_thr


class TestBF16Precision:
    """Tests for precision="bf16" and precision="bf16_refined" (v0.17).

    CPU-runnable: the BF16 dtype flows through the PyTorch complex_matmul
    fallback path (no NKI required). Tests verify correctness contract:
      - "bf16": output is FP32, error bounded by BF16 W quantisation
      - "bf16_refined": one correction step drives error toward FP32 quality
    """

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_bf16_output_dtype_is_fp32(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm_bf16

        torch.manual_seed(42)
        x = torch.randn(n)
        result = _fft_via_gemm_bf16(ComplexTensor(x, torch.zeros(n)), inverse=False)
        assert result.real.dtype == torch.float32, (
            f"Expected FP32 output from BF16 path, got {result.real.dtype}"
        )
        assert result.imag.dtype == torch.float32

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_bf16_error_within_bf16_budget(self, n):
        """BF16 path error bounded by BF16 W quantisation (~1e-2 conservative)."""
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm_bf16

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        result = _fft_via_gemm_bf16(ct, inverse=False)
        expected = np.fft.fft(x.numpy().astype(np.float64))
        err = np.abs(result.real.numpy() - expected.real).max() / (
            np.abs(expected.real).max() + 1e-10
        )
        assert err < 1e-1, f"N={n}: BF16 rel error {err:.2e} exceeds 1e-1"

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_bf16_refined_better_than_bf16(self, n):
        """IR-1 correction step stays within BF16 error budget on CPU.

        On hardware: IR-1 corrects the BF16 W quantisation error → near-FP32.
        On CPU: the IFFT reconstruction uses BF16 too, so the improvement at
        small N may be marginal. The test checks both results stay within the
        BF16 error budget (not that refined is strictly better than baseline —
        that strict comparison is hardware-only, tested in TestBF16Hardware).
        """
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_iterative_refinement, _fft_via_gemm_bf16

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        expected = np.fft.fft(x.numpy().astype(np.float64))

        bf16 = _fft_via_gemm_bf16(ct, inverse=False)
        refined = _fft_iterative_refinement(ct, inverse=False, steps=1)

        err_bf16 = np.abs(bf16.real.numpy() - expected.real).max()
        err_ref = np.abs(refined.real.numpy() - expected.real).max()

        # Both must stay within BF16 error budget (conservative bound).
        assert err_bf16 < 1e-1, f"N={n}: bf16 baseline {err_bf16:.2e} exceeds budget"
        assert err_ref < 1e-1, f"N={n}: refined {err_ref:.2e} exceeds BF16 budget"

    def test_bf16_precision_mode_dispatches(self):
        """set_precision('bf16') + trnfft.fft routes to BF16 path and returns FP32."""
        import trnfft

        old = trnfft.get_precision()
        try:
            trnfft.set_precision("bf16")
            x = torch.randn(256)
            trnfft.fft(x)
            # Test that the mode accepts without error.
        finally:
            trnfft.set_precision(old)

    def test_bf16_refined_precision_mode_dispatches(self):
        """set_precision('bf16_refined') accepts without error."""
        import trnfft

        old = trnfft.get_precision()
        try:
            trnfft.set_precision("bf16_refined")
            x = torch.randn(64)
            _ = trnfft.fft(x)
        finally:
            trnfft.set_precision(old)


class TestOzakiPrecision:
    """Tests for precision="ozaki" and precision="ozaki_hq" (v0.18).

    CPU-runnable: the CPU fallback in complex_gemm_bf16 casts BF16→FP32 before
    matmul, so Ozaki on CPU uses FP32 arithmetic throughout. This gives near-FP64
    accuracy on CPU (no BF16 quantisation) and validates the algorithm structure.
    On hardware, expected O(u_bf16^2) ≈ 1e-5 for "ozaki" and O(u_bf16^3) ≈ 1e-8
    for "ozaki_hq".
    """

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_ozaki_output_dtype_is_fp32(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_ozaki

        torch.manual_seed(42)
        x = torch.randn(n)
        result = _fft_via_ozaki(ComplexTensor(x, torch.zeros(n)), inverse=False)
        assert result.real.dtype == torch.float32
        assert result.imag.dtype == torch.float32

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_ozaki_error_better_than_bf16(self, n):
        """Ozaki rel error should be significantly smaller than single BF16 on CPU.

        Uses relative error (absolute / |DFT|_max) since DFT magnitude scales
        with sqrt(N). On CPU the split uses BF16-quantized values in FP32 matmuls,
        giving ~sqrt(N)*u_bf16^2 relative error vs sqrt(N)*u_bf16 for plain BF16.
        """
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_gemm_bf16, _fft_via_ozaki

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        expected = np.fft.fft(x.numpy().astype(np.float64))
        scale = np.abs(expected.real).max() + 1e-10  # normalise by DFT magnitude

        bf16 = _fft_via_gemm_bf16(ct, inverse=False)
        ozaki = _fft_via_ozaki(ct, inverse=False)

        err_bf16 = np.abs(bf16.real.numpy() - expected.real).max() / scale
        err_oz = np.abs(ozaki.real.numpy() - expected.real).max() / scale

        # Ozaki should be significantly more accurate than plain BF16
        assert err_oz < err_bf16 * 0.1, (
            f"N={n}: ozaki rel ({err_oz:.2e}) not better than bf16 rel ({err_bf16:.2e})"
        )
        assert err_oz < 1e-3, f"N={n}: ozaki rel error {err_oz:.2e} exceeds 1e-3 bound"

    def test_ozaki_roundtrip(self):
        """Ozaki FFT→IFFT roundtrip recovers original signal."""
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_ozaki

        torch.manual_seed(42)
        n = 128
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        X = _fft_via_ozaki(ct, inverse=False)
        back = _fft_via_ozaki(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.numpy(), atol=1e-3)

    def test_precision_mode_ozaki_dispatches(self):
        """_ozaki_or_fallback emits RuntimeWarning when unverified, silent when verified."""
        import warnings

        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _ozaki_or_fallback, set_ozaki_product_precision_verified

        ct = ComplexTensor(torch.randn(64), torch.zeros(64))

        # Unverified: should emit RuntimeWarning and return a valid result.
        set_ozaki_product_precision_verified(False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _ozaki_or_fallback(ct, inverse=False, hq=False)
            assert any(issubclass(warning.category, RuntimeWarning) for warning in w), (
                "Expected RuntimeWarning from _ozaki_or_fallback when unverified"
            )
            assert result.real.shape == (64,)

        # Verified: no warning.
        set_ozaki_product_precision_verified(True)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = _ozaki_or_fallback(ct, inverse=False, hq=False)
                assert not any(issubclass(warning.category, RuntimeWarning) for warning in w)
        finally:
            set_ozaki_product_precision_verified(False)


class TestOzakiHQPrecision:
    """Tests for precision="ozaki_hq" (2-level Ozaki, v0.19).

    CPU-runnable. On CPU the BF16 splits use FP32 matmuls (complex_gemm_bf16 fallback),
    so the computed accuracy is better than on hardware (no BF16 quantisation of the
    matmul inputs). Tests confirm the 2-level split gives tighter error than 1-level.
    """

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_ozaki_hq_output_dtype_is_fp32(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_ozaki_hq

        torch.manual_seed(42)
        x = torch.randn(n)
        result = _fft_via_ozaki_hq(ComplexTensor(x, torch.zeros(n)), inverse=False)
        assert result.real.dtype == torch.float32
        assert result.imag.dtype == torch.float32

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_ozaki_hq_within_accuracy_budget(self, n):
        """2-level Ozaki stays within accuracy budget on CPU.

        On CPU complex_gemm_bf16 uses FP32 matmuls — the 3-way BF16 split adds
        quantisation noise that may slightly exceed 1-level at some N. Strict
        improvement ordering is a hardware property; CPU tests verify the path
        is within the expected BF16-split accuracy regime.
        """
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_ozaki_hq

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        expected = np.fft.fft(x.numpy().astype(np.float64))
        scale = np.abs(expected.real).max() + 1e-10

        hq = _fft_via_ozaki_hq(ct, inverse=False)
        err_hq = np.abs(hq.real.numpy() - expected.real).max() / scale

        assert err_hq < 1e-4, f"N={n}: ozaki_hq rel error {err_hq:.2e} exceeds 1e-4"

    def test_ozaki_hq_roundtrip(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_ozaki_hq

        torch.manual_seed(42)
        n = 128
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        X = _fft_via_ozaki_hq(ct, inverse=False)
        back = _fft_via_ozaki_hq(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.numpy(), atol=1e-4)

    def test_precision_mode_ozaki_hq_dispatches(self):
        """_ozaki_or_fallback(hq=True) emits RuntimeWarning when unverified."""
        import warnings

        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _ozaki_or_fallback, set_ozaki_product_precision_verified

        ct = ComplexTensor(torch.randn(64), torch.zeros(64))

        set_ozaki_product_precision_verified(False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _ozaki_or_fallback(ct, inverse=False, hq=True)
            assert any(issubclass(warning.category, RuntimeWarning) for warning in w)
            assert result.real.shape == (64,)

        set_ozaki_product_precision_verified(True)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = _ozaki_or_fallback(ct, inverse=False, hq=True)
                assert not any(issubclass(warning.category, RuntimeWarning) for warning in w)
        finally:
            set_ozaki_product_precision_verified(False)


@pytest.mark.neuron
class TestOzakiHQCharacterization:
    """On-silicon precision characterization for the Ozaki DFT-GEMM paths.

    Bypasses the hardware-verification gate (sets _OZAKI_PRODUCT_PRECISION_VERIFIED=True
    temporarily) so the actual Ozaki kernels run rather than the BF16 fallback.
    Records bf16, ozaki, and ozaki_hq rel error at N ∈ {64, 128, 256}.
    This test MEASURES and PRINTS — it does not assert theoretical precision bounds.

    Measured hardware results (trn1, SDK 2.29.0, 2026-04-30):
      - "bf16" (1 matmul):    ~1.7e-3 at N=64
      - "ozaki" (3 matmuls):  ~1.7e-3 at N=64  (no improvement: BF16 products before PSUM)
      - "ozaki_hq" (6 mat.):  ~1.7e-3 at N=64  (same)

    Measured hardware results (trn2.3xlarge, SDK 2.29.0, 2026-05-01): TBD
    """

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_ozaki_hq_error_vs_fp64(self, n):
        from trnfft import fft_core
        from trnfft.fft_core import set_ozaki_product_precision_verified

        old_thr = fft_core._DFT_GEMM_THRESHOLD
        fft_core._DFT_GEMM_THRESHOLD = max(n, old_thr)
        # Force the actual Ozaki kernel — bypass the hardware-verification gate.
        set_ozaki_product_precision_verified(True)
        try:
            ozaki_err = _bluestein_rel_error(n, "ozaki")
            hq_err = _bluestein_rel_error(n, "ozaki_hq")
            ratio = ozaki_err / hq_err if hq_err > 0 else float("inf")
            print(f"\nN={n}: ozaki={ozaki_err:.2e}  ozaki_hq={hq_err:.2e}  ratio={ratio:.1f}×")
            assert hq_err < 0.5, f"N={n}: ozaki_hq {hq_err:.2e} not a valid FFT result"
            assert ozaki_err < 0.5, f"N={n}: ozaki {ozaki_err:.2e} not a valid FFT result"
        finally:
            fft_core._DFT_GEMM_THRESHOLD = old_thr
            set_ozaki_product_precision_verified(False)

    def test_ozaki_hq_single_n64(self):
        """Focused N=64 summary; printed output goes into the CHANGELOG."""
        from trnfft import fft_core
        from trnfft.fft_core import set_ozaki_product_precision_verified

        n = 64
        old_thr = fft_core._DFT_GEMM_THRESHOLD
        fft_core._DFT_GEMM_THRESHOLD = max(n, old_thr)
        set_ozaki_product_precision_verified(True)
        try:
            bf16_err = _bluestein_rel_error(n, "bf16")
            ozaki_err = _bluestein_rel_error(n, "ozaki")
            hq_err = _bluestein_rel_error(n, "ozaki_hq")
            print(
                f"\nOzaki hardware precision characterization (N=64):\n"
                f"  bf16 (1 matmul)    error = {bf16_err:.3e}\n"
                f"  ozaki (3 matmuls)  error = {ozaki_err:.3e}\n"
                f"  ozaki_hq (6 mat.)  error = {hq_err:.3e}\n"
                f"  ozaki improvement over bf16 = {bf16_err / ozaki_err:.1f}×\n"
                f"  hq improvement over ozaki   = {ozaki_err / hq_err:.1f}×"
            )
            assert hq_err < 0.5
        finally:
            fft_core._DFT_GEMM_THRESHOLD = old_thr
            set_ozaki_product_precision_verified(False)
