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

    y_cplx = y.real.numpy() + 1j * y.imag.numpy()
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
