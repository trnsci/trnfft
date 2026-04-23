"""Benchmark suite for trnfft on Trainium.

Each operation has up to three baselines:
  * `*_nki`              — trnfft with `set_backend("nki")` (Trainium engines)
  * `*_trnfft_pytorch`   — trnfft with `set_backend("pytorch")` (host CPU)
  * `*_torch`            — vanilla `torch.fft.*` / `torch.matmul` on host CPU

NKI methods are marked `@pytest.mark.neuron` and only run on Trainium hardware.
The other two run wherever PyTorch runs, including on the same trn1 host CPU
when this file is invoked under SSM via `scripts/run_benchmarks.sh`.

NKI benchmarks do an explicit warmup call before timed runs so kernel
compilation cost doesn't pollute steady-state numbers.

Run all benchmarks (saving JSON):
    pytest benchmarks/ -v --benchmark-only --benchmark-json=results.json

Run only PyTorch baselines (no hardware needed):
    pytest benchmarks/ -v -m "not neuron"
"""

from __future__ import annotations

import pytest
import torch

import trnfft
from trnfft import ComplexTensor, complex_matmul
from trnfft.nn import ComplexLinear


def _set(backend: str):
    trnfft.set_backend(backend)


def _warm(fn, *args, **kwargs):
    """Trigger NKI compilation on first call so timed runs see steady state."""
    fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1D FFT
# ---------------------------------------------------------------------------


class TestFFT1D:
    @pytest.mark.neuron
    def test_fft_nki(self, benchmark, random_signal):
        _set("nki")
        try:
            _warm(trnfft.fft, random_signal)
            benchmark(trnfft.fft, random_signal)
        finally:
            _set("auto")

    def test_fft_trnfft_pytorch(self, benchmark, random_signal):
        _set("pytorch")
        try:
            benchmark(trnfft.fft, random_signal)
        finally:
            _set("auto")

    def test_fft_torch(self, benchmark, random_signal):
        benchmark(torch.fft.fft, random_signal)


# ---------------------------------------------------------------------------
# 1D FFT — small-N head-to-head: DFT-as-GEMM vs butterfly
# ---------------------------------------------------------------------------
#
# The DFT-GEMM fast path (trnfft/fft_core.py::_fft_via_gemm) routes FFT
# compute onto the Tensor engine via `complex_gemm`. This benchmark compares
# it head-to-head against the Vector-engine butterfly at sizes where both
# paths are legal, so the Tensor-vs-Vector crossover is empirically observable.
#
# Force path selection by toggling fft_core._DFT_GEMM_THRESHOLD.


@pytest.fixture(params=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
def small_fft_size(request):
    return request.param


@pytest.fixture
def small_random_signal(small_fft_size):
    torch.manual_seed(42)
    return torch.randn(small_fft_size)


class TestFFT1DSmallN:
    """Architectural probe — does DFT-GEMM beat butterfly at small N on Trainium?"""

    @pytest.mark.neuron
    def test_fft_nki_dft_gemm(self, benchmark, small_random_signal):
        from trnfft import fft_core

        old = fft_core._DFT_GEMM_THRESHOLD
        fft_core._DFT_GEMM_THRESHOLD = 1 << 30  # force DFT-GEMM path
        _set("nki")
        try:
            _warm(trnfft.fft, small_random_signal)
            benchmark(trnfft.fft, small_random_signal)
        finally:
            _set("auto")
            fft_core._DFT_GEMM_THRESHOLD = old

    @pytest.mark.neuron
    def test_fft_nki_butterfly(self, benchmark, small_random_signal):
        from trnfft import fft_core

        old = fft_core._DFT_GEMM_THRESHOLD
        fft_core._DFT_GEMM_THRESHOLD = 0  # force butterfly path
        _set("nki")
        try:
            _warm(trnfft.fft, small_random_signal)
            benchmark(trnfft.fft, small_random_signal)
        finally:
            _set("auto")
            fft_core._DFT_GEMM_THRESHOLD = old


# Stockham radix-4 only applies to power-of-4 N. Derived fixture filters
# small_fft_size down to {16, 64, 256, 1024, 2048-> no}, collecting only
# those sizes where the third head-to-head variant is actually valid.
def _is_power_of_four(n):
    return n > 0 and (n & (n - 1)) == 0 and (n.bit_length() & 1) == 1


@pytest.fixture(params=[16, 64, 256, 1024, 4096])
def power_of_four_size(request):
    return request.param


@pytest.fixture
def power_of_four_signal(power_of_four_size):
    torch.manual_seed(42)
    return torch.randn(power_of_four_size)


class TestFFT1DStockham:
    """Three-way head-to-head: DFT-GEMM vs Stockham radix-4 vs butterfly.

    Extends the small-N probe into the regime where DFT-GEMM's FP32 O(N²)
    accumulation blocks it (N >= 512) and where butterfly's log2(N)
    launches dominate (N >= 1024). Stockham lives in the middle: Tensor-
    engine-friendly at any N, precision-safe to N=4096+ (log_4(N)
    accumulation).
    """

    @pytest.mark.neuron
    def test_fft_nki_stockham(self, benchmark, power_of_four_signal):
        from trnfft import fft_core

        old_force = fft_core._FORCE_STOCKHAM
        fft_core._FORCE_STOCKHAM = True
        _set("nki")
        try:
            _warm(trnfft.fft, power_of_four_signal)
            benchmark(trnfft.fft, power_of_four_signal)
        finally:
            _set("auto")
            fft_core._FORCE_STOCKHAM = old_force


@pytest.fixture(params=[8, 64, 512, 4096])
def power_of_eight_size(request):
    return request.param


@pytest.fixture
def power_of_eight_signal(power_of_eight_size):
    torch.manual_seed(42)
    return torch.randn(power_of_eight_size)


class TestFFT1DStockhamR8:
    """Radix-8 Stockham benchmark (Thread B).

    Primary targets:
      N=512: 3 radix-8 stages vs 9 butterfly stages (new coverage).
      N=4096: 4 radix-8 stages vs 6 radix-4 stages (improvement).

    Each stage uses nc_matmul (Tensor engine) for W_8, plus a Vector-engine
    twiddle multiply.  Hardware bench determines if Tensor-engine W_8 plus
    fewer stages outweigh the HBM scratch round-trip per stage.
    """

    @pytest.mark.neuron
    def test_fft_nki_stockham_r8(self, benchmark, power_of_eight_signal):
        from trnfft import fft_core

        old_force = fft_core._FORCE_STOCKHAM_R8
        fft_core._FORCE_STOCKHAM_R8 = True
        _set("nki")
        try:
            _warm(trnfft.fft, power_of_eight_signal)
            benchmark(trnfft.fft, power_of_eight_signal)
        finally:
            _set("auto")
            fft_core._FORCE_STOCKHAM_R8 = old_force


@pytest.fixture(params=[1024, 2048])
def mixed_radix_size(request):
    return request.param


@pytest.fixture
def mixed_radix_signal(mixed_radix_size):
    torch.manual_seed(42)
    return torch.randn(mixed_radix_size)


class TestFFT1DStockhamMixed:
    """Mixed-radix Stockham benchmark (v0.16).

    N=1024: [8,8,4,4] = 4 stages (vs radix-4's 5).
    N=2048: [8,8,8,4] = 4 stages (vs butterfly's 11 — new Stockham coverage).
    """

    @pytest.mark.neuron
    def test_fft_nki_stockham_mixed(self, benchmark, mixed_radix_signal):
        from trnfft import fft_core

        old_force = fft_core._FORCE_STOCKHAM_MIXED
        fft_core._FORCE_STOCKHAM_MIXED = True
        _set("nki")
        try:
            _warm(trnfft.fft, mixed_radix_signal)
            benchmark(trnfft.fft, mixed_radix_signal)
        finally:
            _set("auto")
            fft_core._FORCE_STOCKHAM_MIXED = old_force


@pytest.fixture(params=[64, 128, 256])
def bf16_size(request):
    return request.param


@pytest.fixture
def bf16_signal(bf16_size):
    torch.manual_seed(42)
    return torch.randn(bf16_size)


class TestFFT1DBF16:
    """BF16 DFT-GEMM benchmark (v0.17).

    Measures throughput of the BF16 PSUM-FP32 path vs the FP32 "fast" path.
    Expected: ≈2× speedup on the Tensor Engine for BF16 compute.
    Both paths cover N ∈ {64, 128, 256} (DFT-GEMM threshold).
    """

    @pytest.mark.neuron
    def test_fft_bf16_gemm(self, benchmark, bf16_signal):
        from trnfft import fft_core

        old = fft_core._FORCE_BF16_GEMM
        fft_core._FORCE_BF16_GEMM = True
        _set("nki")
        try:
            _warm(trnfft.fft, bf16_signal)
            benchmark(trnfft.fft, bf16_signal)
        finally:
            _set("auto")
            fft_core._FORCE_BF16_GEMM = old


class TestFFT1DOzaki:
    """Ozaki-scheme benchmark (v0.18): 3 BF16 matmuls, O(u_bf16^2) accuracy.

    Measures cost of the 2-split Ozaki path vs BF16 and FP32 DFT-GEMM.
    Key metric: µs per accuracy decade vs bf16 and fast.
    """

    @pytest.mark.neuron
    def test_fft_ozaki(self, benchmark, bf16_signal):
        from trnfft import fft_core

        old = fft_core._FORCE_OZAKI
        fft_core._FORCE_OZAKI = True
        _set("nki")
        try:
            _warm(trnfft.fft, bf16_signal)
            benchmark(trnfft.fft, bf16_signal)
        finally:
            _set("auto")
            fft_core._FORCE_OZAKI = old


# ---------------------------------------------------------------------------
# 2D FFT
# ---------------------------------------------------------------------------


@pytest.fixture(params=[(64, 64), (256, 256), (1024, 1024)])
def fft2_shape(request):
    return request.param


@pytest.fixture
def fft2_image(fft2_shape):
    torch.manual_seed(42)
    return torch.randn(*fft2_shape)


class TestFFT2D:
    @pytest.mark.neuron
    def test_fft2_nki(self, benchmark, fft2_image):
        _set("nki")
        try:
            _warm(trnfft.fft2, fft2_image)
            benchmark(trnfft.fft2, fft2_image)
        finally:
            _set("auto")

    def test_fft2_trnfft_pytorch(self, benchmark, fft2_image):
        _set("pytorch")
        try:
            benchmark(trnfft.fft2, fft2_image)
        finally:
            _set("auto")

    def test_fft2_torch(self, benchmark, fft2_image):
        benchmark(torch.fft.fft2, fft2_image)


# ---------------------------------------------------------------------------
# 3D FFT
# ---------------------------------------------------------------------------


@pytest.fixture(params=[(8, 16, 16), (32, 64, 64)])
def fftn_shape(request):
    return request.param


@pytest.fixture
def fftn_volume(fftn_shape):
    torch.manual_seed(42)
    return torch.randn(*fftn_shape)


class TestFFTN:
    @pytest.mark.neuron
    def test_fftn_nki(self, benchmark, fftn_volume):
        _set("nki")
        try:
            _warm(trnfft.fftn, fftn_volume)
            benchmark(trnfft.fftn, fftn_volume)
        finally:
            _set("auto")

    def test_fftn_trnfft_pytorch(self, benchmark, fftn_volume):
        _set("pytorch")
        try:
            benchmark(trnfft.fftn, fftn_volume)
        finally:
            _set("auto")

    def test_fftn_torch(self, benchmark, fftn_volume):
        benchmark(torch.fft.fftn, fftn_volume)


# ---------------------------------------------------------------------------
# Batched 1D FFT
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        # Small-N shapes below _DFT_GEMM_THRESHOLD (256) — auto-dispatch to
        # DFT-as-GEMM on NKI. Batched input = one matmul for the whole batch,
        # M=B provides full systolic-array partition utilization.
        (32, 128),
        (32, 256),
        # Above-threshold shapes — still on the butterfly path. Regression
        # baseline for comparing v0.11 (pre-DFT-GEMM) numbers.
        (32, 1024),
        (128, 1024),
    ]
)
def batched_shape(request):
    return request.param


@pytest.fixture
def batched_signal(batched_shape):
    torch.manual_seed(42)
    return torch.randn(*batched_shape)


class TestBatchedFFT:
    @pytest.mark.neuron
    def test_batched_fft_nki(self, benchmark, batched_signal):
        _set("nki")
        try:
            _warm(trnfft.fft, batched_signal)
            benchmark(trnfft.fft, batched_signal)
        finally:
            _set("auto")

    def test_batched_fft_trnfft_pytorch(self, benchmark, batched_signal):
        _set("pytorch")
        try:
            benchmark(trnfft.fft, batched_signal)
        finally:
            _set("auto")

    def test_batched_fft_torch(self, benchmark, batched_signal):
        benchmark(torch.fft.fft, batched_signal)


# ---------------------------------------------------------------------------
# Bluestein (arbitrary-size FFT)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[127, 997, 4097])
def bluestein_size(request):
    return request.param


@pytest.fixture
def bluestein_signal(bluestein_size):
    torch.manual_seed(42)
    return torch.randn(bluestein_size)


class TestBluestein:
    @pytest.mark.neuron
    def test_bluestein_nki(self, benchmark, bluestein_signal):
        _set("nki")
        try:
            _warm(trnfft.fft, bluestein_signal)
            benchmark(trnfft.fft, bluestein_signal)
        finally:
            _set("auto")

    def test_bluestein_trnfft_pytorch(self, benchmark, bluestein_signal):
        _set("pytorch")
        try:
            benchmark(trnfft.fft, bluestein_signal)
        finally:
            _set("auto")

    def test_bluestein_torch(self, benchmark, bluestein_signal):
        benchmark(torch.fft.fft, bluestein_signal)


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------


@pytest.fixture
def waveform():
    torch.manual_seed(42)
    return torch.randn(16000)


# STFT parametrization spans the DFT-GEMM threshold (256). The small-n_fft
# variants flatten to many frames of small-N FFT — the canonical "many FFTs
# at small N" case that should collapse to one matmul under v0.12's fast path.
@pytest.fixture(
    params=[
        (128, 64),  # n_fft=128 → DFT-GEMM
        (256, 128),  # n_fft=256 → DFT-GEMM (exactly at threshold)
        (512, 256),  # n_fft=512 → butterfly (pre-v0.12 baseline)
    ]
)
def stft_config(request):
    return request.param


class TestSTFT:
    @pytest.mark.neuron
    def test_stft_nki(self, benchmark, waveform, stft_config):
        n_fft, hop = stft_config
        _set("nki")
        try:
            _warm(trnfft.stft, waveform, n_fft=n_fft, hop_length=hop)
            benchmark(trnfft.stft, waveform, n_fft=n_fft, hop_length=hop)
        finally:
            _set("auto")

    def test_stft_trnfft_pytorch(self, benchmark, waveform, stft_config):
        n_fft, hop = stft_config
        _set("pytorch")
        try:
            benchmark(trnfft.stft, waveform, n_fft=n_fft, hop_length=hop)
        finally:
            _set("auto")

    def test_stft_torch(self, benchmark, waveform, stft_config):
        n_fft, hop = stft_config
        benchmark(torch.stft, waveform, n_fft=n_fft, hop_length=hop, return_complex=True)


# ---------------------------------------------------------------------------
# Complex GEMM
# ---------------------------------------------------------------------------


@pytest.fixture(params=[128, 256, 512, 1024])
def gemm_size(request):
    return request.param


@pytest.fixture
def gemm_complex_pair(gemm_size):
    torch.manual_seed(42)
    a = ComplexTensor(torch.randn(gemm_size, gemm_size), torch.randn(gemm_size, gemm_size))
    b = ComplexTensor(torch.randn(gemm_size, gemm_size), torch.randn(gemm_size, gemm_size))
    return a, b


class TestComplexGEMM:
    @pytest.mark.neuron
    def test_gemm_nki(self, benchmark, gemm_complex_pair):
        from trnfft.nki import complex_gemm

        a, b = gemm_complex_pair
        _set("nki")
        try:
            _warm(complex_gemm, a, b)
            benchmark(complex_gemm, a, b)
        finally:
            _set("auto")

    def test_gemm_trnfft_pytorch(self, benchmark, gemm_complex_pair):
        a, b = gemm_complex_pair
        benchmark(complex_matmul, a, b)

    def test_gemm_torch_complex64(self, benchmark, gemm_complex_pair):
        a, b = gemm_complex_pair
        a_torch = torch.complex(a.real, a.imag)
        b_torch = torch.complex(b.real, b.imag)
        benchmark(torch.matmul, a_torch, b_torch)


# ---------------------------------------------------------------------------
# ComplexLinear forward
# ---------------------------------------------------------------------------


@pytest.fixture(params=[(128, 256), (512, 1024)])
def linear_shape(request):
    return request.param


@pytest.fixture
def linear_layer_and_input(linear_shape):
    in_features, out_features = linear_shape
    torch.manual_seed(42)
    layer = ComplexLinear(in_features, out_features, bias=True)
    # Batch dim chosen as in_features for a square activation tile.
    x = ComplexTensor(torch.randn(in_features, in_features), torch.randn(in_features, in_features))
    return layer, x


class TestComplexLinear:
    @pytest.mark.neuron
    def test_linear_nki(self, benchmark, linear_layer_and_input):
        layer, x = linear_layer_and_input
        _set("nki")
        try:
            _warm(layer, x)
            benchmark(layer, x)
        finally:
            _set("auto")

    def test_linear_trnfft_pytorch(self, benchmark, linear_layer_and_input):
        layer, x = linear_layer_and_input
        _set("pytorch")
        try:
            benchmark(layer, x)
        finally:
            _set("auto")


# ---------------------------------------------------------------------------
# Complex element-wise multiply (mask apply)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[(64, 32), (256, 128), (1024, 512)])
def mask_shape(request):
    return request.param


@pytest.fixture
def mask_pair(mask_shape):
    torch.manual_seed(42)
    mask = ComplexTensor(torch.randn(*mask_shape), torch.randn(*mask_shape))
    spec = ComplexTensor(torch.randn(*mask_shape), torch.randn(*mask_shape))
    return mask, spec


class TestComplexMask:
    @pytest.mark.neuron
    def test_mask_nki(self, benchmark, mask_pair):
        from trnfft.nki import complex_mask_apply

        mask, spec = mask_pair
        _set("nki")
        try:
            _warm(complex_mask_apply, mask, spec)
            benchmark(complex_mask_apply, mask, spec)
        finally:
            _set("auto")

    def test_mask_trnfft_pytorch(self, benchmark, mask_pair):
        mask, spec = mask_pair
        benchmark(lambda m, s: m * s, mask, spec)
