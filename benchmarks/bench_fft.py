"""Benchmark suite for trnfft vs reference implementations.

Run with:
    pytest benchmarks/ -v

To store results:
    pytest benchmarks/ --benchmark-json=results.json

To compare against baseline:
    pytest benchmarks/ --benchmark-save=baseline
    pytest benchmarks/ --benchmark-compare=baseline

Neuron-marked benchmarks run on Trainium hardware via:
    pytest benchmarks/ -v -m neuron
"""

import pytest
import torch
import trnfft
from trnfft import ComplexTensor, complex_matmul


# --- 1D FFT benchmarks ---

class TestFFT1D:

    def test_fft_trnfft(self, benchmark, random_signal):
        benchmark(trnfft.fft, random_signal)

    def test_fft_torch(self, benchmark, random_signal):
        benchmark(torch.fft.fft, random_signal)


# --- 2D FFT benchmarks ---

class TestFFT2D:

    @pytest.fixture
    def image(self):
        torch.manual_seed(42)
        return torch.randn(256, 256)

    def test_fft2_trnfft(self, benchmark, image):
        benchmark(trnfft.fft2, image)

    def test_fft2_torch(self, benchmark, image):
        benchmark(torch.fft.fft2, image)


# --- STFT benchmarks ---

class TestSTFT:

    @pytest.fixture
    def waveform(self):
        torch.manual_seed(42)
        return torch.randn(16000)

    def test_stft_trnfft(self, benchmark, waveform):
        benchmark(trnfft.stft, waveform, n_fft=512, hop_length=256)

    def test_stft_torch(self, benchmark, waveform):
        benchmark(torch.stft, waveform, n_fft=512, hop_length=256, return_complex=True)


# --- Complex GEMM benchmarks ---

class TestComplexGEMM:

    def test_gemm_trnfft(self, benchmark, random_complex_matrix):
        a = random_complex_matrix
        benchmark(complex_matmul, a, a)


# --- NKI hardware benchmarks (require Trainium) ---

@pytest.mark.neuron
class TestNKI:

    def test_fft_nki(self, benchmark, random_signal):
        trnfft.set_backend("nki")
        try:
            benchmark(trnfft.fft, random_signal)
        finally:
            trnfft.set_backend("auto")

    def test_gemm_nki(self, benchmark, random_complex_matrix):
        from trnfft.nki import complex_gemm
        trnfft.set_backend("nki")
        try:
            a = random_complex_matrix
            benchmark(complex_gemm, a, a)
        finally:
            trnfft.set_backend("auto")
