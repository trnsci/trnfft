"""
Speech enhancement via complex ratio mask (cIRM) estimation.

Demonstrates trnfft's STFT + complex NN layers for a speech enhancement
workflow. This is a minimal training loop — see neuron-complex-ops for the
full Speech Enhancement Arena with multi-model comparison.

Usage:
    python examples/speech_stft.py --demo           # Quick synthetic test
    python examples/speech_stft.py --epochs 50      # Real training
"""

import argparse

import torch
import torch.nn as nn

from trnfft import ComplexTensor, stft
from trnfft.nn import ComplexLinear, ComplexModReLU


class SimpleCIRMNet(nn.Module):
    """Minimal cIRM estimator: STFT → complex linear layers → mask."""

    def __init__(self, n_fft: int = 512, hidden: int = 256):
        super().__init__()
        freq_bins = n_fft // 2 + 1
        self.n_fft = n_fft
        self.layer1 = ComplexLinear(freq_bins, hidden)
        self.act1 = ComplexModReLU(hidden)
        self.layer2 = ComplexLinear(hidden, hidden)
        self.act2 = ComplexModReLU(hidden)
        self.output = ComplexLinear(hidden, freq_bins)

    def forward(self, noisy_spec: ComplexTensor) -> ComplexTensor:
        # noisy_spec: (batch, freq, time)
        # Process each time frame
        batch, freq, time = noisy_spec.shape
        # Reshape to (batch*time, freq) for linear layers
        x_re = noisy_spec.real.permute(0, 2, 1).reshape(-1, freq)
        x_im = noisy_spec.imag.permute(0, 2, 1).reshape(-1, freq)
        x = ComplexTensor(x_re, x_im)

        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        mask = self.output(x)

        # Reshape back to (batch, freq, time)
        mask_re = mask.real.reshape(batch, time, freq).permute(0, 2, 1)
        mask_im = mask.imag.reshape(batch, time, freq).permute(0, 2, 1)
        return ComplexTensor(mask_re, mask_im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Quick synthetic demo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    n_fft = args.n_fft
    hop = args.hop

    # Synthetic data: clean signal + noise
    sr = 16000
    duration = 1.0
    t = torch.arange(0, duration, 1.0 / sr)
    clean = torch.sin(2 * 3.14159 * 440 * t) * 0.5
    noise = torch.randn_like(clean) * 0.1
    noisy = clean + noise

    if args.demo:
        print(f"Signal length: {len(noisy)} samples")
        noisy_spec = stft(noisy, n_fft=n_fft, hop_length=hop)
        clean_spec = stft(clean, n_fft=n_fft, hop_length=hop)
        print(f"STFT shape: {noisy_spec.shape} (freq x time)")
        print(f"Noisy energy: {noisy_spec.abs().sum():.1f}")
        print(f"Clean energy: {clean_spec.abs().sum():.1f}")
        print("Demo complete — STFT works on this backend.")
        return

    # Training
    model = SimpleCIRMNet(n_fft=n_fft)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Batch: repeat signal with different noise realizations
    batch_size = 4
    for epoch in range(args.epochs):
        clean_batch = clean.unsqueeze(0).expand(batch_size, -1)
        noise_batch = torch.randn_like(clean_batch) * 0.1
        noisy_batch = clean_batch + noise_batch

        noisy_spec = stft(noisy_batch, n_fft=n_fft, hop_length=hop)
        clean_spec = stft(clean_batch, n_fft=n_fft, hop_length=hop)

        # Ideal ratio mask
        mask = model(noisy_spec)
        enhanced = mask * noisy_spec

        # MSE loss on real and imaginary parts
        loss = (
            (enhanced.real - clean_spec.real) ** 2 + (enhanced.imag - clean_spec.imag) ** 2
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={loss.item():.6f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
