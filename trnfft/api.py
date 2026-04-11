"""
Public API for trnfft — drop-in replacement for torch.fft on Trainium.

Usage:
    import trnfft

    X = trnfft.fft(x)
    x = trnfft.ifft(X)
    S = trnfft.stft(waveform, n_fft=512, hop_length=256)
"""

from __future__ import annotations

import math
import torch
from typing import Optional

from .complex import ComplexTensor
from .fft_core import fft_core


def fft(input: torch.Tensor | ComplexTensor, n: Optional[int] = None) -> ComplexTensor:
    """1-D discrete Fourier Transform along last dimension."""
    x = _to_complex(input)
    if n is not None:
        x = _resize_last(x, n)
    return fft_core(x, inverse=False)


def ifft(input: torch.Tensor | ComplexTensor, n: Optional[int] = None) -> ComplexTensor:
    """1-D inverse discrete Fourier Transform."""
    x = _to_complex(input)
    if n is not None:
        x = _resize_last(x, n)
    return fft_core(x, inverse=True)


def rfft(input: torch.Tensor, n: Optional[int] = None) -> ComplexTensor:
    """1-D FFT of real signal, returning positive frequencies only."""
    if n is not None:
        input = _resize_real(input, n)
    n = input.shape[-1]
    x = ComplexTensor(input)
    result = fft_core(x, inverse=False)
    half = n // 2 + 1
    return ComplexTensor(result.real[..., :half], result.imag[..., :half])


def irfft(input: ComplexTensor, n: Optional[int] = None) -> torch.Tensor:
    """Inverse FFT of Hermitian-symmetric spectrum, returning real signal."""
    if n is None:
        n = 2 * (input.shape[-1] - 1)

    # Reconstruct full spectrum via Hermitian symmetry
    full_re = torch.zeros(*input.shape[:-1], n, dtype=input.dtype)
    full_im = torch.zeros(*input.shape[:-1], n, dtype=input.dtype)
    half = input.shape[-1]
    full_re[..., :half] = input.real
    full_im[..., :half] = input.imag
    # X[n-k] = conj(X[k]) for k=1..n-half
    if n > 1:
        # Indices n-1, n-2, ..., half correspond to conj of indices 1, 2, ..., n-half
        num_neg = n - half
        full_re[..., half:] = torch.flip(input.real[..., 1:1 + num_neg], dims=[-1])
        full_im[..., half:] = -torch.flip(input.imag[..., 1:1 + num_neg], dims=[-1])

    result = fft_core(ComplexTensor(full_re, full_im), inverse=True)
    return result.real


def fft2(input: torch.Tensor | ComplexTensor, s: Optional[tuple[int, int]] = None) -> ComplexTensor:
    """2-D FFT along last two dimensions."""
    x = _to_complex(input)

    if s is not None:
        x = _resize_last(x, s[1])
        x = x.transpose(-2, -1)
        x = _resize_last(x, s[0])
        x = x.transpose(-2, -1)

    # FFT along last dim (columns)
    shape = x.shape
    n_cols = shape[-1]
    flat = ComplexTensor(x.real.reshape(-1, n_cols), x.imag.reshape(-1, n_cols))
    result = fft_core(flat, inverse=False)
    x = ComplexTensor(result.real.reshape(shape), result.imag.reshape(shape))

    # FFT along second-to-last dim (rows) — transpose, FFT, transpose back
    x = x.transpose(-2, -1)
    shape = x.shape
    n_rows = shape[-1]
    flat = ComplexTensor(x.real.reshape(-1, n_rows), x.imag.reshape(-1, n_rows))
    result = fft_core(flat, inverse=False)
    x = ComplexTensor(result.real.reshape(shape), result.imag.reshape(shape))
    x = x.transpose(-2, -1)

    return x


def stft(
    input: torch.Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool = True,
) -> ComplexTensor:
    """Short-time Fourier Transform. Matches torch.stft signature."""
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, dtype=input.dtype)

    if center:
        pad_amount = n_fft // 2
        # torch.nn.functional.pad with reflect needs 2D+ input
        input = input.unsqueeze(0) if input.dim() == 1 else input
        input = torch.nn.functional.pad(input, (pad_amount, pad_amount), mode=pad_mode)
        input = input.squeeze(0) if input.dim() == 2 and input.shape[0] == 1 else input

    # Frame the signal using unfold (vectorized, no Python loop)
    # unfold(dim, size, step) → (..., num_frames, n_fft)
    frames_tensor = input.unfold(-1, n_fft, hop_length)  # (..., num_frames, n_fft)

    # Apply window
    if win_length < n_fft:
        padded_window = torch.zeros(n_fft, dtype=input.dtype)
        offset = (n_fft - win_length) // 2
        padded_window[offset:offset + win_length] = window
        frames_tensor = frames_tensor * padded_window
    else:
        frames_tensor = frames_tensor * window

    # FFT each frame
    frames_complex = ComplexTensor(frames_tensor)
    # Reshape to 2D for fft_core, then reshape back
    orig_shape = frames_complex.shape
    flat_re = frames_complex.real.reshape(-1, n_fft)
    flat_im = frames_complex.imag.reshape(-1, n_fft)
    flat = ComplexTensor(flat_re, flat_im)
    fft_result = fft_core(flat, inverse=False)
    result_re = fft_result.real.reshape(orig_shape)
    result_im = fft_result.imag.reshape(orig_shape)

    if onesided:
        freq_bins = n_fft // 2 + 1
        result_re = result_re[..., :freq_bins]
        result_im = result_im[..., :freq_bins]

    # Transpose to (..., freq, time) convention
    result_re = result_re.transpose(-2, -1)
    result_im = result_im.transpose(-2, -1)

    if normalized:
        scale = 1.0 / math.sqrt(n_fft)
        result_re = result_re * scale
        result_im = result_im * scale

    return ComplexTensor(result_re, result_im)


# --- Helpers ---

def _to_complex(x) -> ComplexTensor:
    if isinstance(x, ComplexTensor):
        return x
    return ComplexTensor(x)


def _resize_last(x: ComplexTensor, n: int) -> ComplexTensor:
    current = x.shape[-1]
    if current == n:
        return x
    if current < n:
        pad_re = torch.zeros(*x.shape[:-1], n - current, dtype=x.dtype)
        pad_im = torch.zeros(*x.shape[:-1], n - current, dtype=x.dtype)
        return ComplexTensor(
            torch.cat([x.real, pad_re], dim=-1),
            torch.cat([x.imag, pad_im], dim=-1),
        )
    return ComplexTensor(x.real[..., :n], x.imag[..., :n])


def _resize_real(x: torch.Tensor, n: int) -> torch.Tensor:
    current = x.shape[-1]
    if current == n:
        return x
    if current < n:
        pad = torch.zeros(*x.shape[:-1], n - current, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)
    return x[..., :n]
