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

@pytest.fixture(params=[(32, 1024), (128, 1024)])
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


class TestSTFT:
    N_FFT = 512
    HOP = 256

    @pytest.mark.neuron
    def test_stft_nki(self, benchmark, waveform):
        _set("nki")
        try:
            _warm(trnfft.stft, waveform, n_fft=self.N_FFT, hop_length=self.HOP)
            benchmark(trnfft.stft, waveform, n_fft=self.N_FFT, hop_length=self.HOP)
        finally:
            _set("auto")

    def test_stft_trnfft_pytorch(self, benchmark, waveform):
        _set("pytorch")
        try:
            benchmark(trnfft.stft, waveform, n_fft=self.N_FFT, hop_length=self.HOP)
        finally:
            _set("auto")

    def test_stft_torch(self, benchmark, waveform):
        benchmark(torch.stft, waveform, n_fft=self.N_FFT,
                  hop_length=self.HOP, return_complex=True)


# ---------------------------------------------------------------------------
# Complex GEMM
# ---------------------------------------------------------------------------

@pytest.fixture(params=[128, 256, 512, 1024])
def gemm_size(request):
    return request.param


@pytest.fixture
def gemm_complex_pair(gemm_size):
    torch.manual_seed(42)
    a = ComplexTensor(torch.randn(gemm_size, gemm_size),
                      torch.randn(gemm_size, gemm_size))
    b = ComplexTensor(torch.randn(gemm_size, gemm_size),
                      torch.randn(gemm_size, gemm_size))
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
    x = ComplexTensor(torch.randn(in_features, in_features),
                      torch.randn(in_features, in_features))
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
