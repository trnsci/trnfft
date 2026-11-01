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
import operator

import torch

from .complex import ComplexTensor
from .fft_core import fft_core

# --- torch.fft argument conventions ---
#
# Every transform takes torch.fft's ``n`` / ``s``, ``dim`` and ``norm``
# arguments with the same meaning (#123). ``norm`` names which direction is
# scaled: "backward" (the default, ``None``) leaves the forward transform
# unscaled and scales the inverse by 1/n; "forward" is the reverse; "ortho"
# scales both by 1/sqrt(n). ``n`` is the product of the transformed signal
# sizes (the output size for c2r transforms).
#
# A transformed ``dim`` other than the last is swapped to the last axis
# before ``fft_core`` and swapped back after, so the kernels always see the
# same row-major (..., n) layout as a last-axis transform.

_NORM_MODES = (None, "backward", "ortho", "forward")
# hfft / ihfft run irfft / rfft in the opposite direction, so their norm swaps.
_SWAPPED_NORM = {None: "forward", "backward": "forward", "forward": "backward", "ortho": "ortho"}

Dims = int | tuple[int, ...] | list[int] | None
Sizes = tuple[int, ...] | list[int] | torch.Size | None


def fft(
    input: torch.Tensor | ComplexTensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> ComplexTensor:
    """1-D discrete Fourier transform along ``dim``, as ``torch.fft.fft``.

    ``n`` zero-pads or truncates the signal along ``dim`` before the
    transform. ``norm`` is ``"backward"`` (default, no scaling), ``"ortho"``
    (1/sqrt(n)) or ``"forward"`` (1/n).
    """
    return _c2c(input, n, dim, norm, inverse=False)


def ifft(
    input: torch.Tensor | ComplexTensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> ComplexTensor:
    """1-D inverse discrete Fourier transform along ``dim``, as ``torch.fft.ifft``.

    ``norm`` is ``"backward"`` (default, 1/n), ``"ortho"`` (1/sqrt(n)) or
    ``"forward"`` (no scaling).
    """
    return _c2c(input, n, dim, norm, inverse=True)


def rfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> ComplexTensor:
    """1-D FFT of a real signal along ``dim``, as ``torch.fft.rfft``.

    Returns the ``n // 2 + 1`` non-negative-frequency bins along ``dim``.
    """
    _check_norm(norm)
    x = _real_input(input, "rfft")
    d = _canon_dim(dim, x.dim())
    x = _swap_last(x, d)
    if n is not None:
        x = _resize_real(x, _check_n(n))
    n = _check_n(x.shape[-1])
    result = fft_core(ComplexTensor(x), inverse=False)
    half = n // 2 + 1
    y = ComplexTensor(result.real[..., :half], result.imag[..., :half])
    return _swap_last(_apply_norm(y, norm, n, inverse=False), d)


def irfft(
    input: torch.Tensor | ComplexTensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> torch.Tensor:
    """Inverse of :func:`rfft` along ``dim``, as ``torch.fft.irfft``. Returns a real tensor.

    ``n`` is the output length along ``dim`` (default ``2 * (L - 1)`` for
    ``L`` input bins).
    """
    _check_norm(norm)
    x = _to_complex(input)
    d = _canon_dim(dim, x.real.dim())
    x = _swap_last(x, d)
    if n is None:
        n = 2 * (x.shape[-1] - 1)
    _check_n(n)

    # An n-point real signal has n//2 + 1 independent bins: use that many,
    # truncating a longer input or zero-padding a shorter one (as torch.fft /
    # NumPy do). Using the input length instead gave wrong values or a shape
    # error whenever n != 2 * (len - 1) and n != 2 * len - 1 (#104).
    half = n // 2 + 1
    re, im = x.real[..., :half], x.imag[..., :half]
    if re.shape[-1] < half:
        pad = (0, half - re.shape[-1])
        re, im = torch.nn.functional.pad(re, pad), torch.nn.functional.pad(im, pad)

    # Reconstruct full spectrum via Hermitian symmetry
    full_re = torch.zeros(*x.shape[:-1], n, dtype=x.dtype, device=x.device)
    full_im = torch.zeros(*x.shape[:-1], n, dtype=x.dtype, device=x.device)
    full_re[..., :half] = re
    full_im[..., :half] = im
    # X[n-k] = conj(X[k]) for k=1..n-half
    if n > 1:
        # Indices n-1, n-2, ..., half correspond to conj of indices 1, 2, ..., n-half
        num_neg = n - half
        full_re[..., half:] = torch.flip(re[..., 1 : 1 + num_neg], dims=[-1])
        full_im[..., half:] = -torch.flip(im[..., 1 : 1 + num_neg], dims=[-1])

    result = fft_core(ComplexTensor(full_re, full_im), inverse=True)
    return _swap_last(_apply_norm(result.real, norm, n, inverse=True), d)


def hfft(
    input: torch.Tensor | ComplexTensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> torch.Tensor:
    """FFT of a Hermitian-symmetric signal along ``dim``, as ``torch.fft.hfft``.

    ``input`` is the one-sided representation (``n // 2 + 1`` bins used).
    Output is real-valued of length ``n`` (default ``2 * (L - 1)``).

    Implementation: ``irfft(input.conj(), n)`` with the opposite ``norm``
    (``hfft`` is a forward transform, so ``"backward"`` means unscaled).

    Inverse of :func:`ihfft`.
    """
    _check_norm(norm)
    return irfft(_to_complex(input).conj(), n=n, dim=dim, norm=_SWAPPED_NORM[norm])


def ihfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> ComplexTensor:
    """IFFT of a real signal along ``dim``, as ``torch.fft.ihfft``.

    Returns the one-sided Hermitian spectrum of length ``n // 2 + 1``.
    Implementation: ``conj(rfft(input, n))`` with the opposite ``norm``
    (1/n by default).

    Inverse of :func:`hfft`.
    """
    _check_norm(norm)
    x = _real_input(input, "ihfft")
    return rfft(x, n=n, dim=dim, norm=_SWAPPED_NORM[norm]).conj()


def fft2(
    input: torch.Tensor | ComplexTensor,
    s: Sizes = None,
    dim: Dims = (-2, -1),
    norm: str | None = None,
) -> ComplexTensor:
    """2-D FFT, as ``torch.fft.fft2`` (default: the last two dims)."""
    return fftn(input, s=s, dim=dim, norm=norm)


def ifft2(
    input: torch.Tensor | ComplexTensor,
    s: Sizes = None,
    dim: Dims = (-2, -1),
    norm: str | None = None,
) -> ComplexTensor:
    """2-D inverse FFT, as ``torch.fft.ifft2`` (default: the last two dims)."""
    return ifftn(input, s=s, dim=dim, norm=norm)


def fftn(
    input: torch.Tensor | ComplexTensor,
    s: Sizes = None,
    dim: Dims = None,
    norm: str | None = None,
) -> ComplexTensor:
    """N-D FFT, as ``torch.fft.fftn``.

    ``dim`` defaults to the last ``len(s)`` dims if ``s`` is given, else all
    dims. ``s[i] == -1`` keeps the input size along ``dim[i]``.
    """
    return _c2cn(input, s, dim, norm, inverse=False)


def ifftn(
    input: torch.Tensor | ComplexTensor,
    s: Sizes = None,
    dim: Dims = None,
    norm: str | None = None,
) -> ComplexTensor:
    """N-D inverse FFT, as ``torch.fft.ifftn``. Same arguments as :func:`fftn`."""
    return _c2cn(input, s, dim, norm, inverse=True)


def rfft2(
    input: torch.Tensor,
    s: Sizes = None,
    dim: Dims = (-2, -1),
    norm: str | None = None,
) -> ComplexTensor:
    """2-D FFT of a real signal, as ``torch.fft.rfft2``.

    Output shape along ``dim``: ``(s[0], s[1] // 2 + 1)``. Hermitian symmetry
    is applied along the last of ``dim`` only.
    """
    return rfftn(input, s=s, dim=dim, norm=norm)


def irfft2(
    input: torch.Tensor | ComplexTensor,
    s: Sizes = None,
    dim: Dims = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    """Inverse of :func:`rfft2`, as ``torch.fft.irfft2``. Returns a real tensor."""
    return irfftn(input, s=s, dim=dim, norm=norm)


def rfftn(
    input: torch.Tensor,
    s: Sizes = None,
    dim: Dims = None,
    norm: str | None = None,
) -> ComplexTensor:
    """N-D FFT of a real signal, as ``torch.fft.rfftn``.

    Hermitian symmetry is applied only to the last dim in ``dim``; that dim's
    output size is ``s[-1] // 2 + 1`` (or ``input.shape[dim[-1]] // 2 + 1``
    if ``s`` is None).
    """
    _check_norm(norm)
    x = _real_input(input, "rfftn")
    dims, sizes = _canon_s_dim(x.shape, s, dim)
    if not dims:
        raise RuntimeError("rfftn must transform at least one axis")
    last_dim, other_dims = dims[-1], dims[:-1]

    # The real-to-complex step on the last of ``dim`` (rfft handles its s / n).
    y = rfft(x, n=sizes[-1], dim=last_dim)
    n_total = sizes[-1] if sizes[-1] is not None else x.shape[last_dim]

    for size, d in zip(sizes[:-1], other_dims, strict=True):
        if size is not None:
            y = _resize_dim(y, d, size)
    for d in other_dims:
        n_total *= _check_n(y.shape[d])
        y = _fft_along_dim(y, d, inverse=False)

    return _apply_norm(y, norm, n_total, inverse=False)


def irfftn(
    input: torch.Tensor | ComplexTensor,
    s: Sizes = None,
    dim: Dims = None,
    norm: str | None = None,
) -> torch.Tensor:
    """Inverse of :func:`rfftn`, as ``torch.fft.irfftn``. Returns a real tensor.

    The last dim in ``dim`` is Hermitian-reconstructed to full length. If
    ``s`` is None (or ``s[-1] == -1``), its size defaults to
    ``2 * (input.shape[dim[-1]] - 1)``.
    """
    _check_norm(norm)
    x = _to_complex(input)
    dims, sizes = _canon_s_dim(x.shape, s, dim)
    if not dims:
        raise RuntimeError("irfftn must transform at least one axis")
    last_dim, other_dims = dims[-1], dims[:-1]

    for size, d in zip(sizes[:-1], other_dims, strict=True):
        if size is not None:
            x = _resize_dim(x, d, size)
    n_last = sizes[-1] if sizes[-1] is not None else 2 * (x.shape[last_dim] - 1)
    n_total = _check_n(n_last)

    # Inverse complex FFT along the other dims first — keeps the Hermitian
    # structure on the last dim intact so irfft can reconstruct correctly.
    for d in other_dims:
        n_total *= _check_n(x.shape[d])
        x = _fft_along_dim(x, d, inverse=True)

    y = irfft(x, n=n_last, dim=last_dim)
    return _apply_norm(y, norm, n_total, inverse=True)


def hfft2(
    input: torch.Tensor | ComplexTensor,
    s: Sizes = None,
    dim: Dims = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    """2-D FFT of a Hermitian-symmetric signal, as ``torch.fft.hfft2``."""
    return hfftn(input, s=s, dim=dim, norm=norm)


def ihfft2(
    input: torch.Tensor,
    s: Sizes = None,
    dim: Dims = (-2, -1),
    norm: str | None = None,
) -> ComplexTensor:
    """2-D inverse FFT of a real signal (one-sided output), as ``torch.fft.ihfft2``."""
    return ihfftn(input, s=s, dim=dim, norm=norm)


def hfftn(
    input: torch.Tensor | ComplexTensor,
    s: Sizes = None,
    dim: Dims = None,
    norm: str | None = None,
) -> torch.Tensor:
    """N-D FFT of a Hermitian-symmetric signal, as ``torch.fft.hfftn``.

    ``input`` is one-sided along the last of ``dim``. Implementation:
    ``irfftn(input.conj(), s, dim)`` with the opposite ``norm``.
    """
    _check_norm(norm)
    return irfftn(_to_complex(input).conj(), s=s, dim=dim, norm=_SWAPPED_NORM[norm])


def ihfftn(
    input: torch.Tensor,
    s: Sizes = None,
    dim: Dims = None,
    norm: str | None = None,
) -> ComplexTensor:
    """N-D inverse FFT of a real signal, one-sided along the last of ``dim``,
    as ``torch.fft.ihfftn``. Implementation: ``conj(rfftn(input, s, dim))``
    with the opposite ``norm``.
    """
    _check_norm(norm)
    x = _real_input(input, "ihfftn")
    return rfftn(x, s=s, dim=dim, norm=_SWAPPED_NORM[norm]).conj()


# --- Helper functions (torch.fft.fftshift / fftfreq equivalents) ---


def fftshift(input: torch.Tensor | ComplexTensor, dim: Dims = None):
    """Move the zero-frequency bin to the centre, as ``torch.fft.fftshift``.

    Rolls each of ``dim`` (default: all dims) by ``size // 2``. Returns the
    same type as ``input`` (``torch.Tensor`` or ``ComplexTensor``).
    """
    return _roll_half(input, dim, inverse=False)


def ifftshift(input: torch.Tensor | ComplexTensor, dim: Dims = None):
    """Inverse of :func:`fftshift`, as ``torch.fft.ifftshift``."""
    return _roll_half(input, dim, inverse=True)


def fftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Sample frequencies of an ``n``-point FFT, as ``torch.fft.fftfreq``.

    ``[0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)``.
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    k = torch.arange(n, device=device)
    k = torch.where(k < (n + 1) // 2, k, k - n)
    out = k.to(dtype) * (1.0 / (n * d))
    return out.requires_grad_(requires_grad)


def rfftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Sample frequencies of :func:`rfft` of an ``n``-point signal, as
    ``torch.fft.rfftfreq``: ``[0, 1, ..., n // 2] / (d * n)``.
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    out = torch.arange(n // 2 + 1, device=device).to(dtype) * (1.0 / (n * d))
    return out.requires_grad_(requires_grad)


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
    """Short-time Fourier Transform. Takes torch.stft's arguments, except
    ``return_complex``: the result is always a ``ComplexTensor`` of shape
    ``(..., freq, frames)``.

    ``window=None`` uses a Hann window of ``win_length``. ``torch.stft`` uses
    a rectangular window (ones) there, so pass ``window`` explicitly to match
    it; with an explicit window the results agree to FP32 tolerance.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, dtype=input.dtype, device=input.device)

    if center:
        pad_amount = n_fft // 2
        # torch.nn.functional.pad with reflect needs 2D+ input. Squeeze only
        # the dim added here: a caller's (1, L) batch must stay (1, F, T),
        # as torch.stft returns (#114).
        was_1d = input.dim() == 1
        input = input.unsqueeze(0) if was_1d else input
        input = torch.nn.functional.pad(input, (pad_amount, pad_amount), mode=pad_mode)
        input = input.squeeze(0) if was_1d else input

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
        padded_window = torch.zeros(n_fft, dtype=input.dtype, device=window.device)
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
    from .nki.multicore import multi_core_fft

    fft_result = multi_core_fft(flat, inverse=False)
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
    """Inverse Short-time Fourier Transform. Reconstructs signal via overlap-add.

    ``window=None`` uses a Hann window of ``win_length``, as :func:`stft` does
    (``torch.istft`` uses ones). Pass the window used for the forward STFT.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length, dtype=input.dtype, device=input.device)

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
        full_re = torch.zeros(
            *spec_re.shape[:-1], full_n, dtype=spec_re.dtype, device=spec_re.device
        )
        full_im = torch.zeros(
            *spec_im.shape[:-1], full_n, dtype=spec_im.dtype, device=spec_im.device
        )
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
    from .nki.multicore import multi_core_fft

    ifft_result = multi_core_fft(flat, inverse=True)
    frames = ifft_result.real.reshape(*spec_re.shape[:-1], n_fft)

    # Build the window for overlap-add
    if win_length < n_fft:
        padded_window = torch.zeros(n_fft, dtype=window.dtype, device=window.device)
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
    output = torch.zeros(*batch_shape, expected_len, dtype=frames.dtype, device=frames.device)
    window_sum = torch.zeros(expected_len, dtype=frames.dtype, device=frames.device)

    for t in range(num_frames):
        start = t * hop_length
        output[..., start : start + n_fft] += frames[..., t, :] * window
        window_sum[start : start + n_fft] += window**2

    # Where the window sum is large enough, normalize. Where it's near zero
    # (boundary samples), fall back to the raw overlap-add of IFFT frames.
    raw_output = torch.zeros(*batch_shape, expected_len, dtype=frames.dtype, device=frames.device)
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
            pad = torch.zeros(
                *batch_shape, length - current, dtype=output.dtype, device=output.device
            )
            output = torch.cat([output, pad], dim=-1)

    return output


# --- Helpers ---


def _promote_real(x: torch.Tensor) -> torch.Tensor:
    """Promote integer / bool tensors to the default float dtype, as torch.fft does.

    The FFT kernels work in the input dtype; an integer input truncated every
    butterfly (power-of-2 N) or every chirp twiddle (Bluestein), giving
    silently wrong values (#113).
    """
    if x.is_floating_point() or x.is_complex():
        return x
    return x.to(torch.get_default_dtype())


def _to_complex(x) -> ComplexTensor:
    if isinstance(x, ComplexTensor):
        if x.real.is_floating_point():
            return x
        return ComplexTensor(_promote_real(x.real), _promote_real(x.imag))
    return ComplexTensor(_promote_real(x))


def _resize_last(x: ComplexTensor, n: int) -> ComplexTensor:
    current = x.shape[-1]
    if current == n:
        return x
    if current < n:
        pad_re = torch.zeros(*x.shape[:-1], n - current, dtype=x.dtype, device=x.device)
        pad_im = torch.zeros(*x.shape[:-1], n - current, dtype=x.dtype, device=x.device)
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
        pad = torch.zeros(*x.shape[:-1], n - current, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)
    return x[..., :n]


def _check_norm(norm: str | None) -> None:
    if norm not in _NORM_MODES:
        raise RuntimeError(f'Invalid normalization mode: "{norm}"')


def _check_n(n: int) -> int:
    """A signal size must be >= 1, as in torch.fft (a negative ``n`` used to
    slice the input silently, e.g. ``n=-1`` dropped the last sample)."""
    n = operator.index(n)
    if n < 1:
        raise RuntimeError(f"Invalid number of data points ({n}) specified")
    return n


def _canon_dim(dim: int, ndim: int) -> int:
    dim = operator.index(dim)
    if ndim == 0:
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if not -ndim <= dim < ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], "
            f"but got {dim})"
        )
    return dim % ndim


def _canon_s_dim(shape, s: Sizes, dim: Dims) -> tuple[list[int], list[int | None]]:
    """Resolve torch.fft's ``s`` / ``dim`` pair to non-negative dims and sizes.

    ``dim=None`` means the last ``len(s)`` dims if ``s`` is given, else all
    dims. A size of ``None`` (``s`` absent, or ``s[i] == -1``) keeps the
    default size for that dim.
    """
    ndim = len(shape)
    if s is not None:
        s = [operator.index(v) for v in s]
    if dim is None:
        if s is None:
            dims = list(range(ndim))
        else:
            if len(s) > ndim:
                raise RuntimeError(
                    f"Got shape with {len(s)} values but input tensor only has {ndim} dimensions."
                )
            dims = list(range(ndim - len(s), ndim))
    else:
        dim = (dim,) if isinstance(dim, int) else dim
        dims = [_canon_dim(d, ndim) for d in dim]
        if s is not None and len(s) != len(dims):
            raise RuntimeError("When given, dim and shape arguments must have the same length")
    if s is None:
        return dims, [None] * len(dims)
    return dims, [None if v == -1 else _check_n(v) for v in s]


def _apply_norm(x, norm: str | None, n: int, inverse: bool):
    """Rescale from fft_core's convention (forward unscaled, inverse 1/n) to ``norm``."""
    if norm is None or norm == "backward":
        return x
    if norm == "ortho":
        factor = math.sqrt(n)
    else:  # "forward"
        factor = n
    if inverse:
        # fft_core divided by n; "ortho" wants 1/sqrt(n), "forward" none.
        return _map_parts(x, lambda t: t * factor)
    return _map_parts(x, lambda t: t / factor)


def _map_parts(x, f):
    if isinstance(x, ComplexTensor):
        return ComplexTensor(f(x.real), f(x.imag))
    return f(x)


def _swap_last(x, dim: int):
    """Swap ``dim`` with the last axis (contiguous). Self-inverse."""
    ndim = len(x.shape)
    if dim == ndim - 1:
        return x
    if isinstance(x, ComplexTensor):
        return x.transpose(dim, -1)
    return x.transpose(dim, -1).contiguous()


def _real_input(x, name: str) -> torch.Tensor:
    if isinstance(x, ComplexTensor) or torch.is_complex(x):
        raise TypeError(f"{name} expects a real input tensor, but got {_dtype_name(x)}")
    return _promote_real(x)


def _dtype_name(x) -> str:
    if isinstance(x, ComplexTensor):
        return f"ComplexTensor({x.dtype})"
    return str(x.dtype)


def _c2c(input, n: int | None, dim: int, norm: str | None, inverse: bool) -> ComplexTensor:
    _check_norm(norm)
    x = _to_complex(input)
    d = _canon_dim(dim, x.real.dim())
    x = _swap_last(x, d)
    if n is not None:
        x = _resize_last(x, _check_n(n))
    n = _check_n(x.shape[-1])
    y = fft_core(x, inverse=inverse)
    return _swap_last(_apply_norm(y, norm, n, inverse), d)


def _c2cn(input, s: Sizes, dim: Dims, norm: str | None, inverse: bool) -> ComplexTensor:
    _check_norm(norm)
    x = _to_complex(input)
    dims, sizes = _canon_s_dim(x.shape, s, dim)
    for size, d in zip(sizes, dims, strict=True):
        if size is not None:
            x = _resize_dim(x, d, size)

    if not dims:
        # No transformed dims: return a copy, never a view of the caller's input,
        # as torch.fft.fftn does (#116).
        return x.clone()

    n_total = 1
    for d in dims:
        n_total *= _check_n(x.shape[d])
        x = _fft_along_dim(x, d, inverse=inverse)
    return _apply_norm(x, norm, n_total, inverse)


def _roll_half(input, dim: Dims, inverse: bool):
    t = input.real if isinstance(input, ComplexTensor) else input
    ndim = t.dim()
    if dim is None:
        dims = list(range(ndim))
    else:
        dim = (dim,) if isinstance(dim, int) else dim
        dims = [_canon_dim(d, ndim) for d in dim]
    if not dims:
        return input.clone()
    shifts = [(-(t.shape[d] // 2)) if inverse else t.shape[d] // 2 for d in dims]
    return _map_parts(input, lambda p: torch.roll(p, shifts, dims))
