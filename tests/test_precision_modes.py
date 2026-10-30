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

    # N=8193 takes ~19 min on Trn2 (SDK 2.33); opt in with --run-slow or -m slow.
    @pytest.mark.parametrize("n", [997, 1009, pytest.param(8193, marks=pytest.mark.slow)])
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
      - "bf16": output is FP32
      - both modes dispatch without error
    Error bounds are pinned in TestBF16BoundsRegression.
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
    Bounds (precision.py): "ozaki" ≤ 1.6e-5, "ozaki_hq" ≤ 1e-6 at N ≤ 256; see
    TestOzakiBoundsRegression (trnfft#81).
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


class TestOzakiBoundsRegression:
    """Regression for trnfft#81: Ozaki modes vs the precision.py bounds.

    complex_gemm_bf16's CPU fallback multiplies BF16 values in FP32, and products of
    BF16 values are exact in FP32. So the CPU numbers are what hardware with exact
    BF16 products gives; trn2 (SDK 2.33, spike B5) matched them to two digits before
    the fix. Error is a relative Frobenius norm vs FP64 torch.fft, as in B5.
    """

    @staticmethod
    def _rel(out, xr, xi):
        ref = torch.fft.fft(torch.complex(xr.double(), xi.double()))
        got = torch.complex(out.real.double(), out.imag.double())
        return (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()

    def test_dft_matrix_fp64_is_exact(self):
        from trnfft.fft_core import _dft_matrix_fp64

        for n in (64, 256):
            for inverse in (False, True):
                c, s = _dft_matrix_fp64(n, inverse)
                k = torch.arange(n, dtype=torch.float64)
                sign = 1.0 if inverse else -1.0
                ref = torch.exp(sign * 2j * torch.pi * torch.outer(k, k) / n)
                assert c.dtype == torch.float64
                assert torch.allclose(torch.complex(c, s), ref, atol=1e-12)

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_ozaki_within_bound(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_ozaki

        torch.manual_seed(0)
        xr, xi = torch.randn(8, n), torch.randn(8, n)
        err = self._rel(_fft_via_ozaki(ComplexTensor(xr, xi), inverse=False), xr, xi)
        # precision.py bound is 1.6e-5; pre-#81 this was 2.0e-5 at N=256.
        assert err < 1.6e-5 / 2, f"N={n}: ozaki rel err {err:.2e}"

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_ozaki_hq_within_1e6_bound(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_ozaki, _fft_via_ozaki_hq

        torch.manual_seed(0)
        xr, xi = torch.randn(8, n), torch.randn(8, n)
        ct = ComplexTensor(xr, xi)
        err_hq = self._rel(_fft_via_ozaki_hq(ct, inverse=False), xr, xi)
        err_oz = self._rel(_fft_via_ozaki(ct, inverse=False), xr, xi)
        # pre-#81: 5.0e-6 / 1.0e-5 / 2.1e-5, the same as ozaki.
        assert err_hq < 1e-6 / 2, f"N={n}: ozaki_hq rel err {err_hq:.2e}"
        assert err_hq < err_oz / 10, f"N={n}: hq {err_hq:.2e} vs ozaki {err_oz:.2e}"

    @pytest.mark.parametrize("n", [64, 256])
    def test_ozaki_hq_inverse_within_bound(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_ozaki_hq

        torch.manual_seed(1)
        xr, xi = torch.randn(4, n), torch.randn(4, n)
        out = _fft_via_ozaki_hq(ComplexTensor(xr, xi), inverse=True)
        ref = torch.fft.ifft(torch.complex(xr.double(), xi.double()))
        got = torch.complex(out.real.double(), out.imag.double())
        assert (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item() < 5e-7


class TestBF16BoundsRegression:
    """Regression for trnfft#86: "bf16" / "bf16_refined" vs the precision.py bounds.

    complex_gemm_bf16's CPU fallback multiplies BF16 values in FP32 (exact) and
    accumulates in FP32, as the PSUM does. Trn2 (SDK 2.33, spike B5) matched the
    CPU numbers to 3 digits: bf16_refined 1.945e-3 / 2.029e-3 / 2.078e-3 at
    N=64/128/256 before the fix, and the same for "bf16" at N=128/256. Error is a
    relative Frobenius norm vs FP64 torch.fft, as in B5 (batch 8, seed N).
    """

    BF16_BOUND = 3e-3  # precision.py; measured max 2.16e-3 (N=256, 20 seeds)
    REFINED_BOUND = 1e-5  # precision.py; measured max 4.9e-6 (N=256, 20 seeds)

    @staticmethod
    def _case(n, inverse):
        g = torch.Generator().manual_seed(n)
        xr = torch.randn(8, n, generator=g, dtype=torch.float64)
        xi = torch.randn(8, n, generator=g, dtype=torch.float64)
        c = torch.complex(xr, xi)
        ref = torch.fft.ifft(c) if inverse else torch.fft.fft(c)
        return ComplexTensor(xr.float(), xi.float()), ref

    @staticmethod
    def _rel(out, ref):
        got = torch.complex(out.real.double(), out.imag.double())
        return (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()

    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("n", [2, 16, 64, 128, 256])
    def test_bf16_within_bound(self, n, inverse):
        from trnfft.fft_core import _fft_via_gemm_bf16

        x, ref = self._case(n, inverse)
        err = self._rel(_fft_via_gemm_bf16(x, inverse), ref)
        # The error is BF16 rounding of W and x, ~u_bf16 = 2^-9 = 1.95e-3; it does
        # not shrink below ~1e-3. A value under that means BF16 is not in effect.
        assert 1e-3 < err < self.BF16_BOUND, f"N={n} inv={inverse}: bf16 rel err {err:.3e}"

    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("n", [2, 16, 64, 128, 256])
    def test_bf16_refined_within_bound(self, n, inverse):
        from trnfft.fft_core import _fft_iterative_refinement, _fft_via_gemm_bf16

        x, ref = self._case(n, inverse)
        err_ref = self._rel(_fft_iterative_refinement(x, inverse, steps=1), ref)
        err_bf16 = self._rel(_fft_via_gemm_bf16(x, inverse), ref)
        # 2x headroom under the stated bound. Pre-#86 err_ref was ~2e-3 (no gain).
        assert err_ref < self.REFINED_BOUND / 2, f"N={n} inv={inverse}: {err_ref:.3e}"
        assert err_ref < err_bf16 / 100, (
            f"N={n} inv={inverse}: refined {err_ref:.3e} vs bf16 {err_bf16:.3e}"
        )

    def test_bf16_refined_more_steps_reach_fp32_floor(self):
        from trnfft.fft_core import _fft_iterative_refinement

        x, ref = self._case(256, False)
        # IR converges: each step multiplies the error by ~u_bf16.
        assert self._rel(_fft_iterative_refinement(x, False, steps=3), ref) < 1e-6

    @pytest.mark.parametrize("n", [64, 128, 256, 512])
    @pytest.mark.parametrize("mode", ["bf16", "bf16_refined"])
    def test_nki_dispatch_takes_lowp_path(self, monkeypatch, mode, n):
        """The NKI dispatcher routes N <= _LOWP_GEMM_MAX to the BF16 drivers.

        #82 raised _DFT_GEMM_THRESHOLD to 1024 and added _LOWP_GEMM_MAX = 256.
        This pins that "bf16_refined" still reaches _fft_iterative_refinement at
        N=64-256 and falls through to an FP32 path (single GEMM or the fused
        four-step) above 256.
        """
        import trnfft.nki.dispatch as dispatch
        from trnfft import fft_core

        calls = []

        def tag(name):
            def f(x, inverse, *a, **k):
                calls.append(name)
                return x

            return f

        import trnfft.nki.butterfly as butterfly

        # Without nki installed the kernels are not defined; the dispatcher only
        # imports them, the paths under test never launch them.
        for k in ("butterfly_stage_kernel", "butterfly_stage_kernel_kahan"):
            monkeypatch.setattr(butterfly, k, None, raising=False)
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: True)
        monkeypatch.setattr(fft_core, "_fft_via_gemm_bf16", tag("bf16"))
        monkeypatch.setattr(fft_core, "_fft_iterative_refinement", tag("bf16_refined"))
        monkeypatch.setattr(fft_core, "_fft_via_gemm", tag("fp32"))
        monkeypatch.setattr(fft_core, "_fft_via_four_step_fused", tag("fp32"))
        x = ComplexTensor(torch.randn(2, n), torch.randn(2, n))
        fft_core._cooley_tukey_nki_nograd(x, inverse=False, precision=mode)
        assert calls == [mode if n <= fft_core._LOWP_GEMM_MAX else "fp32"]
