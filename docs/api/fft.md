# FFT Operations

All functions are available at the top level: `trnfft.fft()`, `trnfft.ifft()`, etc.

## Common arguments

Every transform takes `torch.fft`'s size, `dim` and `norm` arguments with the same meaning (#123):

- **n** (1-D) / **s** (2-D, N-D): signal size along each transformed dim. The input is zero-padded or truncated to it. For the c2r transforms (`irfft*`, `hfft*`) this is the *output* size along the last transformed dim (default `2 * (L - 1)` for `L` input bins), and the first `n // 2 + 1` input bins are used. `s[i] == -1` keeps the default size for that dim. Sizes below 1 raise `RuntimeError`.
- **dim**: the dim to transform (1-D, default `-1`) or the dims (2-D default `(-2, -1)`; N-D default: the last `len(s)` dims if `s` is given, else all dims). Negative dims count from the end. For `rfft*` / `irfft*` / `hfft*` / `ihfft*` the one-sided (Hermitian) dim is the *last* entry of `dim`.
- **norm**: `None` / `"backward"` (default), `"ortho"` or `"forward"`. For a size-`n` transform (`n` = product of the transformed sizes):

  | norm | forward (`fft`, `rfft`, `hfft`, ...) | inverse (`ifft`, `irfft`, `ihfft`, ...) |
  |---|---|---|
  | `"backward"` | no scaling | `1/n` |
  | `"ortho"` | `1/sqrt(n)` | `1/sqrt(n)` |
  | `"forward"` | `1/n` | no scaling |

  The default is unchanged from earlier releases. An unknown mode raises `RuntimeError`.

Inputs: complex transforms take a `torch.Tensor` (real or complex) or a `ComplexTensor`; `rfft*` / `ihfft*` take a real `torch.Tensor` (complex input raises `TypeError`). Integer and bool inputs are promoted to the default float dtype. Complex outputs are `ComplexTensor`; real outputs are `torch.Tensor`.

**Non-last dims and the NKI path.** A transformed dim other than the last is swapped to the last axis (a contiguous copy) before the transform and swapped back after, so the NKI kernels see the same `(..., n)` row layout as a last-axis transform. Strided loads that avoid the copy are future work (#123).

**Differences from `torch.fft`:** no `out=` argument. A repeated entry in `dim` transforms that dim twice, where `torch.fft` raises (#127). `fftshift(x, dim=())` returns a copy, where `torch.fft.fftshift` raises.

## 1-D transforms

### `fft(input, n=None, dim=-1, norm=None)` / `ifft(input, n=None, dim=-1, norm=None)`

1-D DFT and inverse DFT along `dim`. Returns a `ComplexTensor`.

### `rfft(input, n=None, dim=-1, norm=None)`

1-D FFT of a real signal, returning only the `n // 2 + 1` non-negative frequencies along `dim`.

### `irfft(input, n=None, dim=-1, norm=None)`

Inverse of `rfft`. Reconstructs the full spectrum via Hermitian symmetry and returns a real `torch.Tensor` of length `n` along `dim` (default `2 * (L - 1)`). Uses the first `n // 2 + 1` input bins (truncated or zero-padded), as `torch.fft.irfft` does; before v0.26.0 every bin was used, which was wrong for most `n` (#105).

### `hfft(input, n=None, dim=-1, norm=None)` / `ihfft(input, n=None, dim=-1, norm=None)`

FFT of a Hermitian-symmetric signal (real output of length `n`) and its inverse (one-sided spectrum of `n // 2 + 1` bins). `hfft` is a forward transform, so its default `norm` leaves it unscaled and `ihfft` scales by `1/n`. Since v0.16.0; `hfft` accepts a real input since #123.

## 2-D and N-D transforms

### `fft2` / `ifft2` / `rfft2` / `irfft2` / `hfft2` / `ihfft2` `(input, s=None, dim=(-2, -1), norm=None)`

2-D transforms; each delegates to its N-D version. `ifft2`, `hfft2` and `ihfft2` are new in #123.

### `fftn` / `ifftn` `(input, s=None, dim=None, norm=None)`

N-D DFT and inverse along `dim`. `dim=()` returns a copy of the input.

### `rfftn` / `irfftn` `(input, s=None, dim=None, norm=None)`

N-D real-to-complex FFT and its inverse. Only the last dim in `dim` is one-sided (`s[-1] // 2 + 1` bins); `irfftn` returns a real tensor.

### `hfftn` / `ihfftn` `(input, s=None, dim=None, norm=None)`

N-D FFT of a signal that is Hermitian-symmetric along the last of `dim` (real output), and its inverse (one-sided along the last of `dim`). New in #123.

## Helper functions

### `fftshift(input, dim=None)` / `ifftshift(input, dim=None)`

Roll each dim in `dim` (default: all) by `size // 2` (`fftshift`) or `-(size // 2)` (`ifftshift`), moving the zero-frequency bin to the centre and back. Takes and returns a `torch.Tensor` or a `ComplexTensor`.

### `fftfreq(n, d=1.0, *, dtype=None, device=None, requires_grad=False)` / `rfftfreq(...)`

Sample frequencies of an `n`-point `fft` (`[0, 1, ..., (n-1)//2, -(n//2), ..., -1] / (d*n)`) and `rfft` (`[0, ..., n//2] / (d*n)`), as `torch.fft.fftfreq` / `rfftfreq`. `out=` and `layout=` are not supported.

## Time-frequency

### `stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode="reflect", normalized=False, onesided=True)`

Short-time Fourier Transform. Takes `torch.stft`'s arguments except `return_complex` (the result is always a `ComplexTensor`).

- **window**: Default `None` means a **Hann** window of `win_length`. `torch.stft` uses a rectangular window (ones) there, so pass `window` explicitly to match it.

- **Returns**: `ComplexTensor` with shape `(..., freq_bins, num_frames)`

### `istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=True, length=None)`

Inverse STFT via overlap-add reconstruction. `window=None` is a Hann window, as in `stft`; pass the window used for the forward transform.

- **length**: Exact output length (truncates or pads).
- **Returns**: `torch.Tensor` (real-valued)

## Algorithms

- **Power-of-2 sizes**: Cooley-Tukey radix-2 (iterative, decimation-in-time) on the PyTorch backend. On the NKI path (FP32): a single DFT-GEMM for N < 512, the batch-fused four-step (two GEMMs, one launch per batch) up to N = 4096, and the six-step (N = N1·N2, two batched four-step passes, all on the device) for N ≥ 8192. The six-step is the default since v0.25.0; `TRNFFT_SIX_STEP=0` restores the Stockham / butterfly path and `TRNFFT_SIX_STEP_MIN` moves the threshold.
- **Arbitrary sizes**: Bluestein's chirp-z transform (pads to power-of-2, 3 FFTs)
- **float64**: on the NKI path, float64 input runs the FP32 kernels and returns float64; `precision="double"` computes on the host in FP64 (v0.26.0, #109).
- **Plan caching**: Plans are cached by `(size, inverse)` — first call computes, subsequent calls reuse.
