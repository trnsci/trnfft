# FFT Operations

All functions are available at the top level: `trnfft.fft()`, `trnfft.ifft()`, etc.

## 1D transforms

### `fft(input, n=None)`

1-D discrete Fourier Transform along the last dimension.

- **input**: `torch.Tensor` or `ComplexTensor`
- **n**: Output size (zero-pads or truncates input). Default: input size.
- **Returns**: `ComplexTensor`

### `ifft(input, n=None)`

1-D inverse DFT. Same signature as `fft`.

### `rfft(input, n=None)`

1-D FFT of a real signal, returning only positive frequencies (N//2 + 1 bins).

- **input**: `torch.Tensor` (real-valued)
- **Returns**: `ComplexTensor` with shape `(..., N//2 + 1)`

### `irfft(input, n=None)`

Inverse of `rfft`. Reconstructs full spectrum via Hermitian symmetry.

- **input**: `ComplexTensor` (positive frequencies)
- **n**: Output signal length. Default: `2 * (input.shape[-1] - 1)`. Uses the first `n // 2 + 1` input bins (truncated or zero-padded), as `torch.fft.irfft` does; before v0.26.0 every bin was used, which was wrong for most `n` (#105).
- **Returns**: `torch.Tensor` (real-valued)

### `hfft(input, n=None)` / `ihfft(input, n=None)`

FFT of a Hermitian-symmetric signal (real output of length `n`) and its inverse (one-sided spectrum of `n // 2 + 1` bins). Since v0.16.0.

## N-D transforms

### `fft2(input, s=None)`

2-D FFT along the last two dimensions. Delegates to `fftn`.

### `fftn(input, s=None, dim=None)`

N-D FFT along specified dimensions.

- **s**: Output sizes per dimension (tuple)
- **dim**: Dimensions to transform (tuple). Default: all dimensions.

### `ifftn(input, s=None, dim=None)`

N-D inverse FFT. Same signature as `fftn`.

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
