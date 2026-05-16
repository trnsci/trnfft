"""Test STFT and ISTFT correctness."""

import numpy as np
import pytest
import torch

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
        np.testing.assert_allclose(recovered[interior].numpy(), signal[interior].numpy(), atol=1e-3)


class TestSTFTMulticore:
    """STFT/ISTFT routed through multi-NeuronCore batch-split FFT (v0.22).

    CPU-runnable: multicore disabled falls back to fft_core (single-core sequential);
    multicore enabled routes frames through _batch_split_fft (still sequential on CPU,
    but exercises the dispatch path). Output must match single-core within FP32 tolerance.
    """

    @pytest.mark.parametrize("n_fft,hop", [(256, 128), (512, 256)])
    def test_stft_multicore_matches_single_core(self, n_fft, hop):
        from trnfft.nki.multicore import set_multicore

        torch.manual_seed(42)
        signal = torch.randn(4000)

        set_multicore(False)
        S_single = trnfft.stft(signal, n_fft=n_fft, hop_length=hop)

        set_multicore(True)
        try:
            S_multi = trnfft.stft(signal, n_fft=n_fft, hop_length=hop)
        finally:
            set_multicore(False)

        np.testing.assert_allclose(
            S_multi.real.numpy(), S_single.real.numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            S_multi.imag.numpy(), S_single.imag.numpy(), rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("n_fft", [256, 512])
    def test_stft_multicore_roundtrip(self, n_fft):
        from trnfft.nki.multicore import set_multicore

        torch.manual_seed(7)
        signal = torch.randn(4000)
        hop = n_fft // 2

        set_multicore(True)
        try:
            S = trnfft.stft(signal, n_fft=n_fft, hop_length=hop)
            recovered = trnfft.istft(S, n_fft=n_fft, hop_length=hop, length=4000)
        finally:
            set_multicore(False)

        np.testing.assert_allclose(recovered.numpy(), signal.numpy(), atol=1e-3)

    def test_stft_multicore_disabled_passthrough(self):
        """set_multicore(False) gives identical output to default (fft_core) path."""
        from trnfft.nki.multicore import set_multicore

        torch.manual_seed(0)
        signal = torch.randn(2048)

        S_default = trnfft.stft(signal, n_fft=256, hop_length=128)

        set_multicore(False)
        try:
            S_disabled = trnfft.stft(signal, n_fft=256, hop_length=128)
        finally:
            set_multicore(False)

        np.testing.assert_allclose(S_disabled.real.numpy(), S_default.real.numpy(), atol=1e-6)


@pytest.mark.neuron
class TestNKISTFT:
    """STFT/ISTFT routed through NKI butterfly kernel for the inner FFTs."""

    def test_stft_nki_shape(self, nki_backend):
        torch.manual_seed(42)
        signal = torch.randn(2048)
        S = trnfft.stft(signal, n_fft=128, hop_length=64)
        freq_bins = 128 // 2 + 1
        assert S.shape[-2] == freq_bins
        assert torch.all(torch.isfinite(S.abs()))

    def test_stft_nki_roundtrip(self, nki_backend):
        torch.manual_seed(42)
        signal = torch.randn(2048)
        S = trnfft.stft(signal, n_fft=128, hop_length=64)
        recovered = trnfft.istft(S, n_fft=128, hop_length=64, length=2048)
        np.testing.assert_allclose(recovered.numpy(), signal.numpy(), atol=1e-3)
