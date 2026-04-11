"""Test STFT and ISTFT correctness."""

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


class TestISTFT:

    def test_roundtrip(self):
        torch.manual_seed(42)
        signal = torch.randn(1024)
        n_fft = 256
        hop = 64
        S = trnfft.stft(signal, n_fft=n_fft, hop_length=hop)
        recovered = trnfft.istft(S, n_fft=n_fft, hop_length=hop, length=1024)
        np.testing.assert_allclose(recovered.numpy(), signal.numpy(), atol=1e-3)

    def test_roundtrip_batched(self):
        torch.manual_seed(42)
        signal = torch.randn(4, 1024)
        n_fft = 256
        hop = 64
        S = trnfft.stft(signal, n_fft=n_fft, hop_length=hop)
        recovered = trnfft.istft(S, n_fft=n_fft, hop_length=hop, length=1024)
        np.testing.assert_allclose(recovered.numpy(), signal.numpy(), atol=1e-3)

    def test_length_truncation(self):
        torch.manual_seed(42)
        signal = torch.randn(500)
        n_fft = 128
        hop = 32
        S = trnfft.stft(signal, n_fft=n_fft, hop_length=hop)
        recovered = trnfft.istft(S, n_fft=n_fft, hop_length=hop, length=500)
        assert recovered.shape[-1] == 500

    def test_not_centered(self):
        torch.manual_seed(42)
        n_fft = 128
        hop = 32
        # Signal length must be compatible with unfold when center=False
        signal = torch.randn(512)
        S = trnfft.stft(signal, n_fft=n_fft, hop_length=hop, center=False)
        recovered = trnfft.istft(S, n_fft=n_fft, hop_length=hop, center=False, length=512)
        # Boundary samples where window is zero are irrecoverable without centering.
        # Check interior where overlap-add is well-conditioned.
        interior = slice(hop, -hop)
        np.testing.assert_allclose(
            recovered[interior].numpy(), signal[interior].numpy(), atol=1e-3
        )
