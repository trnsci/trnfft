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

from .complex import ComplexTensor
from .fft_core import fft_core


def fft(input: torch.Tensor | ComplexTensor, n: int | None = None) -> ComplexTensor:
    """1-D discrete Fourier Transform along last dimension."""
    x = _to_complex(input)
    if n is not None:
        x = _resize_last(x, n)
    return fft_core(x, inverse=False)


def ifft(input: torch.Tensor | ComplexTensor, n: int | None = None) -> ComplexTensor:
    """1-D inverse discrete Fourier Transform."""
    x = _to_complex(input)
    if n is not None:
        x = _resize_last(x, n)
    return fft_core(x, inverse=True)


def rfft(input: torch.Tensor, n: int | None = None) -> ComplexTensor:
    """1-D FFT of real signal, returning positive frequencies only."""
    if n is not None:
        input = _resize_real(input, n)
    n = input.shape[-1]
    x = ComplexTensor(input)
    result = fft_core(x, inverse=False)
    half = n // 2 + 1
    return ComplexTensor(result.real[..., :half], result.imag[..., :half])


def irfft(input: ComplexTensor, n: int | None = None) -> torch.Tensor:
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
        full_re[..., half:] = torch.flip(input.real[..., 1 : 1 + num_neg], dims=[-1])
        full_im[..., half:] = -torch.flip(input.imag[..., 1 : 1 + num_neg], dims=[-1])

    result = fft_core(ComplexTensor(full_re, full_im), inverse=True)
    return result.real


def hfft(input: ComplexTensor, n: int | None = None) -> torch.Tensor:
    """FFT of a Hermitian-symmetric signal; returns a real frequency-domain output.

    ``input`` is the one-sided representation of length ``n//2+1``. Output is
    real-valued of length ``n``.

    Implementation: ``irfft(input.conj(), n) * n``. This matches NumPy/PyTorch
    semantics: hfft uses the "forward" norm convention for irfft (no 1/n factor),
    so the standard irfft result (which divides by n) must be multiplied back by n.

    Inverse of :func:`ihfft`.
    """
    if n is None:
        n = 2 * (input.shape[-1] - 1)
    conj_input = ComplexTensor(input.real, -input.imag)
    return irfft(conj_input, n=n) * n


def ihfft(input: torch.Tensor, n: int | None = None) -> ComplexTensor:
    """IFFT of a real signal; returns one-sided Hermitian spectrum of length n//2+1.

    Implementation: ``rfft(input, n) / n``. This matches NumPy/PyTorch semantics:
    ihfft uses the "forward" norm convention for rfft (1/n scaling on the forward
    transform), so the standard rfft result is divided by n.

    Inverse of :func:`hfft`.
    """
    n_actual = input.shape[-1] if n is None else n
    result = rfft(input, n=n_actual)
    # ihfft uses the "forward" (opposite-direction) rfft convention:
    # conj(rfft(x)) / n — imaginary part is negated relative to rfft.
    return ComplexTensor(result.real / n_actual, -result.imag / n_actual)


def fft2(input: torch.Tensor | ComplexTensor, s: tuple[int, int] | None = None) -> ComplexTensor:
    """2-D FFT along last two dimensions."""
    s_arg = None if s is None else tuple(s)
    return fftn(input, s=s_arg, dim=(-2, -1))


def fftn(
    input: torch.Tensor | ComplexTensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
) -> ComplexTensor:
    """N-D FFT along specified dimensions (default: all)."""
    x = _to_complex(input)
    ndim = len(x.shape)

    if dim is None:
        if s is not None:
            dim = tuple(range(-len(s), 0))
        else:
            dim = tuple(range(-ndim, 0))

    # Normalize negative dims
    dim = tuple(d % ndim for d in dim)

    # Resize if s is provided
    if s is not None:
        assert len(s) == len(dim), f"len(s)={len(s)} must match len(dim)={len(dim)}"
        for size, d in zip(s, dim, strict=True):
            x = _resize_dim(x, d, size)

    # Apply 1D FFT along each dimension
    for d in dim:
        x = _fft_along_dim(x, d, inverse=False)

    return x


def ifftn(
    input: torch.Tensor | ComplexTensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
) -> ComplexTensor:
    """N-D inverse FFT along specified dimensions (default: all)."""
    x = _to_complex(input)
    ndim = len(x.shape)

    if dim is None:
        if s is not None:
            dim = tuple(range(-len(s), 0))
        else:
            dim = tuple(range(-ndim, 0))

    dim = tuple(d % ndim for d in dim)

    if s is not None:
        assert len(s) == len(dim), f"len(s)={len(s)} must match len(dim)={len(dim)}"
        for size, d in zip(s, dim, strict=True):
            x = _resize_dim(x, d, size)

    for d in dim:
        x = _fft_along_dim(x, d, inverse=True)

    return x


def rfft2(
    input: torch.Tensor,
    s: tuple[int, int] | None = None,
) -> ComplexTensor:
    """2-D FFT of a real signal along the last two dimensions.

    Output shape: `(..., s[0], s[1] // 2 + 1)` — Hermitian symmetry along the
    last dim only. If `s` is None, uses the input's last-two-dim shape.
    """
    return rfftn(input, s=tuple(s) if s is not None else None, dim=(-2, -1))


def rfftn(
    input: torch.Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
) -> ComplexTensor:
    """N-D FFT of a real signal along specified dimensions.

    Hermitian symmetry is applied only to the last dim in ``dim``; that dim's
    output size is ``s[-1] // 2 + 1`` (or ``input.shape[last_dim] // 2 + 1``
    if ``s`` is None).
    """
    assert not torch.is_complex(input), "rfftn requires real input"
    ndim = input.dim()

    if dim is None:
        if s is not None:
            dim = tuple(range(-len(s), 0))
        else:
            dim = tuple(range(-ndim, 0))
    dim = tuple(d % ndim for d in dim)

    last_dim = dim[-1]
    other_dims = dim[:-1]

    # rfft acts on the *last* tensor dim. If the caller's last real-axis dim
    # isn't the last tensor dim, transpose first.
    n_last = s[-1] if s is not None else None
    if last_dim == ndim - 1:
        x = rfft(input, n=n_last)
    else:
        x_t = input.transpose(last_dim, -1).contiguous()
        x = rfft(x_t, n=n_last)
        x = x.transpose(last_dim, -1)

    # Resize the other dims per s (the last-dim resize was handled by rfft's n=).
    if s is not None:
        for size, d in zip(s[:-1], other_dims, strict=True):
            x = _resize_dim(x, d, size)

    # Apply full complex FFT along the remaining dims.
    for d in other_dims:
        x = _fft_along_dim(x, d, inverse=False)

    return x


def irfft2(
    input: ComplexTensor,
    s: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Inverse of :func:`rfft2`. Returns a real tensor along the last two dims.

    If `s` is None, infers the last-dim output size as ``2 * (N_half - 1)``
    where ``N_half = input.shape[-1]``, and the second-to-last dim from the
    input shape.
    """
    s_arg = tuple(s) if s is not None else None
    return irfftn(input, s=s_arg, dim=(-2, -1))


def irfftn(
    input: ComplexTensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Inverse of :func:`rfftn`. Returns a real tensor.

    The last dim in ``dim`` is Hermitian-reconstructed to full length. If
    ``s`` is None, its size defaults to ``2 * (input.shape[last_dim] - 1)``.
    """
    ndim = input.real.dim()

    if dim is None:
        if s is not None:
            dim = tuple(range(-len(s), 0))
        else:
            dim = tuple(range(-ndim, 0))
    dim = tuple(d % ndim for d in dim)

    last_dim = dim[-1]
    other_dims = dim[:-1]

    # Resize the non-last dims per s (the real-axis size is handled via irfft's n=).
    x = input
    if s is not None:
        for size, d in zip(s[:-1], other_dims, strict=True):
            x = _resize_dim(x, d, size)
        n_last = s[-1]
    else:
        n_last = 2 * (x.shape[last_dim] - 1)

    # Inverse complex FFT along the other dims first — keeps the Hermitian
    # structure on the last dim intact so irfft can reconstruct correctly.
    for d in other_dims:
        x = _fft_along_dim(x, d, inverse=True)

    # Final step: irfft along the last_dim. irfft only acts on the *last*
    # tensor dim; transpose if needed, then transpose the real result back.
    if last_dim == ndim - 1:
        return irfft(x, n=n_last)
    # Move last_dim to the end, irfft, move back.
    x_t = ComplexTensor(
        x.real.transpose(last_dim, -1).contiguous(),
        x.imag.transpose(last_dim, -1).contiguous(),
    )
    y = irfft(x_t, n=n_last)
    return y.transpose(last_dim, -1).contiguous()


def stft(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
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

    # Frame the signal. torch.Tensor.unfold is a view op that is not
    # implemented on the XLA backend (torch-xla 2.9 / torch-neuronx), so we
    # build the frame index matrix explicitly — this works on CPU, CUDA, MPS,
    # and XLA/Trainium without a device-specific branch.
    padded_length = input.shape[-1]
    num_frames = 1 + (padded_length - n_fft) // hop_length
    frame_idx = (
        torch.arange(n_fft, device=input.device).unsqueeze(0)
        + torch.arange(num_frames, device=input.device).unsqueeze(1) * hop_length
    )
    frames_tensor = input[..., frame_idx]  # (..., num_frames, n_fft)

    # Apply window
    if win_length < n_fft:
        padded_window = torch.zeros(n_fft, dtype=input.dtype)
        offset = (n_fft - win_length) // 2
        padded_window[offset : offset + win_length] = window
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


def istft(
    input: ComplexTensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = True,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse Short-time Fourier Transform. Reconstructs signal via overlap-add."""
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, dtype=input.dtype)

    # Undo normalization
    if normalized:
        scale = math.sqrt(n_fft)
        input = ComplexTensor(input.real * scale, input.imag * scale)

    # Input is (..., freq, time) — transpose to (..., time, freq)
    spec_re = input.real.transpose(-2, -1)
    spec_im = input.imag.transpose(-2, -1)

    # Reconstruct full spectrum if onesided
    if onesided:
        freq_bins = spec_re.shape[-1]
        full_n = n_fft
        full_re = torch.zeros(*spec_re.shape[:-1], full_n, dtype=spec_re.dtype)
        full_im = torch.zeros(*spec_im.shape[:-1], full_n, dtype=spec_im.dtype)
        full_re[..., :freq_bins] = spec_re
        full_im[..., :freq_bins] = spec_im
        if full_n > 1:
            num_neg = full_n - freq_bins
            full_re[..., freq_bins:] = torch.flip(spec_re[..., 1 : 1 + num_neg], dims=[-1])
            full_im[..., freq_bins:] = -torch.flip(spec_im[..., 1 : 1 + num_neg], dims=[-1])
        spec_re = full_re
        spec_im = full_im

    # IFFT each frame
    num_frames = spec_re.shape[-2]
    flat_re = spec_re.reshape(-1, n_fft)
    flat_im = spec_im.reshape(-1, n_fft)
    flat = ComplexTensor(flat_re, flat_im)
    ifft_result = fft_core(flat, inverse=True)
    frames = ifft_result.real.reshape(*spec_re.shape[:-1], n_fft)

    # Build the window for overlap-add
    if win_length < n_fft:
        padded_window = torch.zeros(n_fft, dtype=window.dtype)
        offset = (n_fft - win_length) // 2
        padded_window[offset : offset + win_length] = window
        window = padded_window

    # Overlap-add with window normalization
    # The analysis window was applied in stft(). For perfect reconstruction,
    # we apply the synthesis window and divide by the sum of squared windows.
    # At boundaries where window_sum is near zero, we use the unnormalized
    # IFFT output directly (no window weighting can recover those samples).
    expected_len = n_fft + (num_frames - 1) * hop_length
    batch_shape = frames.shape[:-2]
    output = torch.zeros(*batch_shape, expected_len, dtype=frames.dtype)
    window_sum = torch.zeros(expected_len, dtype=frames.dtype)

    for t in range(num_frames):
        start = t * hop_length
        output[..., start : start + n_fft] += frames[..., t, :] * window
        window_sum[start : start + n_fft] += window**2

    # Where the window sum is large enough, normalize. Where it's near zero
    # (boundary samples), fall back to the raw overlap-add of IFFT frames.
    raw_output = torch.zeros(*batch_shape, expected_len, dtype=frames.dtype)
    for t in range(num_frames):
        start = t * hop_length
        raw_output[..., start : start + n_fft] += frames[..., t, :]

    mask = window_sum > 1e-8
    output[..., mask] = output[..., mask] / window_sum[mask]
    output[..., ~mask] = raw_output[..., ~mask]

    # Remove center padding
    if center:
        pad_amount = n_fft // 2
        output = output[..., pad_amount:]
        if output.shape[-1] > pad_amount:
            output = output[..., :-pad_amount] if length is None else output

    # Trim or pad to requested length
    if length is not None:
        current = output.shape[-1]
        if current > length:
            output = output[..., :length]
        elif current < length:
            pad = torch.zeros(*batch_shape, length - current, dtype=output.dtype)
            output = torch.cat([output, pad], dim=-1)

    return output


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


def _fft_along_dim(x: ComplexTensor, dim: int, inverse: bool) -> ComplexTensor:
    """Apply 1D FFT along a specific dimension by moving it to the last position."""
    ndim = len(x.shape)
    dim = dim % ndim

    # Move target dim to last position
    if dim != ndim - 1:
        x = x.transpose(dim, -1)

    # Flatten to 2D, FFT, reshape back
    shape = x.shape
    n = shape[-1]
    flat = ComplexTensor(x.real.reshape(-1, n), x.imag.reshape(-1, n))
    result = fft_core(flat, inverse=inverse)
    x = ComplexTensor(result.real.reshape(shape), result.imag.reshape(shape))

    # Move back
    if dim != ndim - 1:
        x = x.transpose(dim, -1)

    return x


def _resize_dim(x: ComplexTensor, dim: int, n: int) -> ComplexTensor:
    """Resize ComplexTensor along a specific dimension."""
    ndim = len(x.shape)
    dim = dim % ndim
    if dim != ndim - 1:
        x = x.transpose(dim, -1)
    x = _resize_last(x, n)
    if dim != ndim - 1:
        x = x.transpose(dim, -1)
    return x


def _resize_real(x: torch.Tensor, n: int) -> torch.Tensor:
    current = x.shape[-1]
    if current == n:
        return x
    if current < n:
        pad = torch.zeros(*x.shape[:-1], n - current, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)
    return x[..., :n]
