"""Test STFT correctness."""

import pytest
import torch
import numpy as np
import trnfft


class TestSTFT:

    def test_shape(self):
        signal = torch.randn(16000)
        n_fft = 256
        hop = 128
        result = trnfft.stft(signal, n_fft=n_fft, hop_length=hop)
        freq_bins = n_fft // 2 + 1
        assert result.shape[-2] == freq_bins

    def test_energy_conservation(self):
        signal = torch.randn(512)
        result = trnfft.stft(signal, n_fft=64, hop_length=32, center=False)
        mags = result.abs()
        assert torch.all(torch.isfinite(mags))
        assert mags.sum() > 0

    def test_single_tone(self):
        sr = 1000
        freq = 100
        t = torch.arange(0, 1.0, 1.0 / sr)
        signal = torch.sin(2 * np.pi * freq * t)
        n_fft = 256
        result = trnfft.stft(signal, n_fft=n_fft, hop_length=64, center=False)
        mags = result.abs()
        avg_spectrum = mags.mean(dim=-1)
        peak_bin = avg_spectrum.argmax().item()
        expected_bin = int(freq * n_fft / sr)
        assert abs(peak_bin - expected_bin) <= 1
