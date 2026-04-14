"""Simulator-backed kernel correctness tests (NKI 0.3.0 Stable).

Run with ``TRNFFT_USE_SIMULATOR=1`` on any x86_64 Linux host that has
``nki>=0.3.0`` installed. Bypasses torch_xla + NEFF compile; routes kernel
dispatch through ``nki.simulate(kernel)(np_args)``.

Intentionally curated to small shapes — the CPU simulator is slow at 1024
and above. Correctness parity with hardware at these scales is what we're
verifying, not perf. Catches Python-trace-level errors (bad kwargs,
dropped ops, shape mismatches); MLIR verifier errors remain hardware-only.
"""

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nki_simulator


@pytest.fixture(autouse=True)
def _simulator_enabled():
    """Skip the whole module if TRNFFT_USE_SIMULATOR isn't set.

    The marker alone isn't sufficient — users may ``pytest -m nki_simulator``
    on a host where nki isn't importable or the env var hasn't been set.
    Fail loudly vs silently falling back.
    """
    if os.environ.get("TRNFFT_USE_SIMULATOR", "").lower() not in ("1", "true", "yes"):
        pytest.skip("TRNFFT_USE_SIMULATOR=1 required")

    from trnfft.nki.dispatch import HAS_NKI

    if not HAS_NKI:
        pytest.skip("nki package not importable on this host")


class TestComplexGEMMSimulator:
    def test_aligned_128(self):
        import trnfft
        from trnfft import ComplexTensor
        from trnfft.nki.dispatch import complex_gemm

        trnfft.set_backend("nki")
        try:
            torch.manual_seed(0)
            a = ComplexTensor(torch.randn(128, 128), torch.randn(128, 128))
            b = ComplexTensor(torch.randn(128, 128), torch.randn(128, 128))
            c = complex_gemm(a, b)
            expected_re = a.real @ b.real - a.imag @ b.imag
            expected_im = a.real @ b.imag + a.imag @ b.real
            torch.testing.assert_close(c.real, expected_re, atol=1e-3, rtol=1e-4)
            torch.testing.assert_close(c.imag, expected_im, atol=1e-3, rtol=1e-4)
        finally:
            trnfft.set_backend("auto")


class TestComplexMulSimulator:
    def test_divisible_by_128(self):
        import trnfft
        from trnfft import ComplexTensor
        from trnfft.nki.dispatch import complex_mask_apply

        trnfft.set_backend("nki")
        try:
            torch.manual_seed(0)
            a = ComplexTensor(torch.randn(128, 4), torch.randn(128, 4))
            b = ComplexTensor(torch.randn(128, 4), torch.randn(128, 4))
            c = complex_mask_apply(a, b)
            expected_re = a.real * b.real - a.imag * b.imag
            expected_im = a.real * b.imag + a.imag * b.real
            torch.testing.assert_close(c.real, expected_re, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(c.imag, expected_im, atol=1e-4, rtol=1e-4)
        finally:
            trnfft.set_backend("auto")


class TestStockhamSimulator:
    """Radix-4 Stockham FFT kernel under nki.simulate.

    Validates the NKI port agrees with the CPU reference in
    ``trnfft.stockham`` to FP32 rtol. Curated to small power-of-4 sizes
    so the simulator finishes in reasonable CI time — perf question is
    hardware-only.
    """

    @pytest.mark.parametrize("n", [64, 256])
    def test_stockham_nki_matches_cpu(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_stockham_nki
        from trnfft.stockham import stockham_radix4

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        sim = _fft_via_stockham_nki(ct, inverse=False)
        cpu = stockham_radix4(ct, inverse=False)
        np.testing.assert_allclose(sim.real.numpy(), cpu.real.numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(sim.imag.numpy(), cpu.imag.numpy(), atol=1e-4, rtol=1e-4)

    def test_stockham_nki_matches_numpy_n256(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_stockham_nki

        torch.manual_seed(42)
        x = torch.randn(256)
        sim = _fft_via_stockham_nki(ComplexTensor(x, torch.zeros(256)), inverse=False)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(sim.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(sim.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)


class TestFFTSimulator:
    """FFT routes through butterfly_stage_kernel over log2(N) stages under
    nki.simulate. Small N only — simulator is not fast at 1024+."""

    @pytest.mark.parametrize("n", [512, 1024])
    def test_fft_vs_numpy(self, n):
        import trnfft

        trnfft.set_backend("nki")
        try:
            torch.manual_seed(0)
            x = torch.randn(n)
            result = trnfft.fft(x)
            expected = np.fft.fft(x.numpy())
            np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
            np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)
        finally:
            trnfft.set_backend("auto")
