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

    @pytest.mark.parametrize(
        "driver,n",
        [
            ("_fft_via_stockham_nki", 1024),
            ("_fft_via_stockham_nki_r8", 512),
            ("_fft_via_stockham_nki_mixed", 2048),
        ],
    )
    def test_row_chunked_bit_identical(self, monkeypatch, driver, n):
        """trnfft#83: launches capped at 128 rows match one launch per stage."""
        from trnfft import fft_core as fc
        from trnfft.complex import ComplexTensor

        torch.manual_seed(83)
        x = ComplexTensor(torch.randn(2, n), torch.randn(2, n))
        fn = getattr(fc, driver)
        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", str(1 << 30))
        whole = fn(x, inverse=False)
        monkeypatch.setenv("TRNFFT_MAX_ROWS_PER_LAUNCH", "128")
        chunked = fn(x, inverse=False)
        assert torch.equal(chunked.real, whole.real)
        assert torch.equal(chunked.imag, whole.imag)

    def test_stockham_nki_matches_numpy_n256(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_stockham_nki

        torch.manual_seed(42)
        x = torch.randn(256)
        sim = _fft_via_stockham_nki(ComplexTensor(x, torch.zeros(256)), inverse=False)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(sim.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(sim.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)

    def test_fft_auto_routes_to_stockham_at_n1024(self):
        """At N=1024 power-of-4, trnfft.fft() dispatches to Stockham (256 <
        N ≤ 4096 with power-of-4 shape)."""
        import trnfft

        trnfft.set_backend("nki")
        try:
            torch.manual_seed(42)
            x = torch.randn(1024)
            result = trnfft.fft(x)
            expected = np.fft.fft(x.numpy())
            np.testing.assert_allclose(result.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
            np.testing.assert_allclose(result.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)
        finally:
            trnfft.set_backend("auto")


class TestStockhamR8Simulator:
    """Radix-8 Stockham kernel under nki.simulate (Thread B).

    Validates the NKI port (twiddle Vector + W_8 Tensor nc_matmul) agrees
    with the CPU reference. N=8 and N=64 only — simulator is slow at 512+.
    """

    @pytest.mark.parametrize("n", [8, 64])
    def test_stockham_r8_nki_matches_cpu(self, n):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_stockham_nki_r8
        from trnfft.stockham import stockham_radix8

        torch.manual_seed(42)
        x = torch.randn(n)
        ct = ComplexTensor(x, torch.zeros(n))
        sim = _fft_via_stockham_nki_r8(ct, inverse=False)
        cpu = stockham_radix8(ct, inverse=False)
        np.testing.assert_allclose(sim.real.numpy(), cpu.real.numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(sim.imag.numpy(), cpu.imag.numpy(), atol=1e-4, rtol=1e-4)

    def test_stockham_r8_nki_matches_numpy_n64(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_stockham_nki_r8

        torch.manual_seed(42)
        x = torch.randn(64)
        sim = _fft_via_stockham_nki_r8(ComplexTensor(x, torch.zeros(64)), inverse=False)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(sim.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(sim.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)

    def test_inverse_r8_roundtrip(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_stockham_nki_r8

        torch.manual_seed(42)
        x = torch.randn(64)
        ct = ComplexTensor(x, torch.zeros(64))
        X = _fft_via_stockham_nki_r8(ct, inverse=False)
        back = _fft_via_stockham_nki_r8(X, inverse=True)
        np.testing.assert_allclose(back.real.numpy(), x.numpy(), atol=1e-4)
        np.testing.assert_allclose(back.imag.numpy(), np.zeros(64), atol=1e-4)


class TestStockhamMixedSimulator:
    """Mixed-radix Stockham driver under nki.simulate (v0.16).

    N=64 → plan=[8,4] (2 stages: one r8 + one r4). Small enough for simulator.
    Validates that the two-kernel interleaving in _fft_via_stockham_nki_mixed
    produces results matching the CPU reference.
    """

    def test_mixed_nki_matches_cpu_n64(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_stockham_nki_mixed
        from trnfft.stockham import stockham_mixed_radix

        torch.manual_seed(42)
        x = torch.randn(64)
        ct = ComplexTensor(x, torch.zeros(64))
        sim = _fft_via_stockham_nki_mixed(ct, inverse=False)
        cpu = stockham_mixed_radix(ct, inverse=False)
        np.testing.assert_allclose(sim.real.numpy(), cpu.real.numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(sim.imag.numpy(), cpu.imag.numpy(), atol=1e-4, rtol=1e-4)

    def test_mixed_nki_matches_numpy_n64(self):
        from trnfft.complex import ComplexTensor
        from trnfft.fft_core import _fft_via_stockham_nki_mixed

        torch.manual_seed(42)
        x = torch.randn(64)
        sim = _fft_via_stockham_nki_mixed(ComplexTensor(x, torch.zeros(64)), inverse=False)
        expected = np.fft.fft(x.numpy())
        np.testing.assert_allclose(sim.real.numpy(), expected.real, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(sim.imag.numpy(), expected.imag, atol=1e-3, rtol=1e-3)


class TestFFTSimulator:
    """FFT routes through butterfly_stage_kernel over log2(N) stages under
    nki.simulate. Small N only — simulator is not fast at 1024+."""

    @pytest.mark.parametrize("n", [512, 1024])
    def test_fft_vs_numpy(self, n):
        # N=512 routes to radix-8 Stockham (3 stages); N=1024 routes to butterfly.
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


class TestFourStepFusedSimulator:
    """Batch-fused four-step kernel vs the unfused driver and numpy (trnfft#82)."""

    @pytest.mark.parametrize("n,batch", [(256, 5), (512, 24), (2048, 9), (4096, 32)])
    @pytest.mark.parametrize("inverse", [False, True])
    def test_matches_unfused_and_numpy(self, n, batch, inverse):
        from trnfft import ComplexTensor
        from trnfft.fft_core import _fft_via_four_step, _fft_via_four_step_fused

        g = torch.Generator().manual_seed(n + batch)
        xr, xi = torch.randn(batch, n, generator=g), torch.randn(batch, n, generator=g)
        x = ComplexTensor(xr, xi)
        fused = _fft_via_four_step_fused(x, inverse)
        unfused = _fft_via_four_step(x, inverse)
        xc = xr.double().numpy() + 1j * xi.double().numpy()
        ref = np.fft.ifft(xc) if inverse else np.fft.fft(xc)
        got = fused.real.double().numpy() + 1j * fused.imag.double().numpy()
        assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-6
        scale = unfused.real.abs().max()
        assert (fused.real - unfused.real).abs().max() / scale < 1e-6
        assert (fused.imag - unfused.imag).abs().max() / scale < 1e-6

    def test_ragged_batch_matches_numpy(self):
        # B=38 at N=4096: packs of 2 rows, a 3-pack tail group. The unfused
        # driver's twiddle kernel cannot take this batch (B*N/128 = 1216).
        from trnfft import ComplexTensor
        from trnfft.fft_core import _fft_via_four_step_fused

        g = torch.Generator().manual_seed(38)
        xr, xi = torch.randn(38, 4096, generator=g), torch.randn(38, 4096, generator=g)
        y = _fft_via_four_step_fused(ComplexTensor(xr, xi), False)
        ref = np.fft.fft(xr.double().numpy() + 1j * xi.double().numpy())
        got = y.real.double().numpy() + 1j * y.imag.double().numpy()
        assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-6


class TestBF16GEMMSimulator:
    """BF16 complex GEMM kernel smoke test under nki.simulate (v0.17).

    Validates that _complex_gemm_kernel_bf16 compiles and runs in the NKI
    simulator with BF16 inputs and produces FP32 output. Small N only.
    """

    def test_bf16_kernel_output_is_fp32(self):
        """Kernel returns FP32 output tensors for BF16 inputs."""
        from trnfft.nki.dispatch import _complex_gemm_kernel_bf16, _simulate_kernel

        torch.manual_seed(42)
        n = 64
        a = torch.randn(1, n, dtype=torch.bfloat16)
        a_i = torch.zeros(1, n, dtype=torch.bfloat16)
        b = torch.randn(n, n, dtype=torch.bfloat16)
        b_i = torch.zeros(n, n, dtype=torch.bfloat16)

        c_r, c_i = _simulate_kernel(_complex_gemm_kernel_bf16, a, a_i, b, b_i)
        assert c_r.dtype == torch.float32, f"Expected FP32 output, got {c_r.dtype}"
        assert c_i.dtype == torch.float32

    def test_bf16_gemm_matches_fp32_within_bf16_budget(self):
        """BF16 GEMM result agrees with FP32 reference within BF16 quantisation budget."""
        from trnfft.complex import ComplexTensor, complex_matmul
        from trnfft.nki.dispatch import _complex_gemm_kernel_bf16, _simulate_kernel

        torch.manual_seed(42)
        m, k, n = 128, 64, 64
        a_r = torch.randn(m, k, dtype=torch.bfloat16)
        a_i = torch.zeros(m, k, dtype=torch.bfloat16)
        b_r = torch.randn(k, n, dtype=torch.bfloat16)
        b_i = torch.zeros(k, n, dtype=torch.bfloat16)

        # Simulator result
        c_r, c_i = _simulate_kernel(_complex_gemm_kernel_bf16, a_r, a_i, b_r, b_i)

        # FP32 reference with BF16-quantised inputs (simulates FP32 PSUM)
        ref = complex_matmul(
            ComplexTensor(a_r.float(), a_i.float()),
            ComplexTensor(b_r.float(), b_i.float()),
        )
        np.testing.assert_allclose(
            c_r.numpy(),
            ref.real.numpy(),
            atol=1e-1,
            rtol=1e-1,
            err_msg="BF16 GEMM sim result diverges from FP32 reference beyond BF16 budget",
        )


class TestComplexGemmTNSimulator:
    """_complex_gemm_tn_kernel (trnfft#97) under nki.simulate: stacked blocks
    given transposed give the same bits as _complex_gemm_kernel on the
    stacked untransposed blocks."""

    @pytest.mark.parametrize("g,k,m,n", [(3, 64, 128, 64), (2, 32, 64, 32), (1, 256, 256, 1024)])
    def test_matches_gemm_kernel_bitwise(self, g, k, m, n):
        from trnfft.nki.dispatch import (
            _complex_gemm_kernel,
            _complex_gemm_tn_kernel,
            _simulate_kernel,
        )

        gen = torch.Generator().manual_seed(g * k + m)
        at_r, at_i = torch.randn(g * k, m, generator=gen), torch.randn(g * k, m, generator=gen)
        b_r, b_i = torch.randn(k, n, generator=gen), torch.randn(k, n, generator=gen)
        c_r, c_i = _simulate_kernel(_complex_gemm_tn_kernel, at_r, at_i, b_r, b_i)
        a_r = at_r.reshape(g, k, m).transpose(1, 2).reshape(g * m, k).contiguous()
        a_i = at_i.reshape(g, k, m).transpose(1, 2).reshape(g * m, k).contiguous()
        w_r, w_i = _simulate_kernel(_complex_gemm_kernel, a_r, a_i, b_r, b_i)
        assert c_r.shape == (g * m, n)
        assert torch.equal(c_r, w_r) and torch.equal(c_i, w_i)


class TestPETransposeSimulator:
    """trnfft#97: the plain-load + Tensor-engine-transpose kernels give the same
    bits as the ``load_transpose2d`` kernels they replace."""

    @pytest.mark.parametrize("m,k,n", [(1, 8, 8), (16, 64, 64), (256, 128, 128), (256, 256, 1024)])
    def test_gemm_matches_t2d_bitwise(self, m, k, n):
        from trnfft.nki.dispatch import (
            _complex_gemm_kernel,
            _complex_gemm_t2d_kernel,
            _simulate_kernel,
        )

        gen = torch.Generator().manual_seed(m * k + n)
        args = [torch.randn(m, k, generator=gen), torch.randn(m, k, generator=gen)]
        args += [torch.randn(k, n, generator=gen), torch.randn(k, n, generator=gen)]
        c_r, c_i = _simulate_kernel(_complex_gemm_kernel, *args)
        w_r, w_i = _simulate_kernel(_complex_gemm_t2d_kernel, *args)
        assert torch.equal(c_r, w_r) and torch.equal(c_i, w_i)

    @pytest.mark.parametrize("groups", [64, 256])
    def test_radix8_matches_t2d_bitwise(self, groups):
        from trnfft.nki.dispatch import _simulate_kernel
        from trnfft.nki.stockham import stockham_radix8_w8_kernel, stockham_radix8_w8_t2d_kernel

        gen = torch.Generator().manual_seed(groups)
        args = [torch.randn(groups, 8, generator=gen), torch.randn(groups, 8, generator=gen)]
        args += [torch.randn(8, 8, generator=gen), torch.randn(8, 8, generator=gen)]
        c_r, c_i = _simulate_kernel(stockham_radix8_w8_kernel, *args)
        w_r, w_i = _simulate_kernel(stockham_radix8_w8_t2d_kernel, *args)
        assert torch.equal(c_r, w_r) and torch.equal(c_i, w_i)
