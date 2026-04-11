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
- **n**: Output signal length. Default: `2 * (input.shape[-1] - 1)`
- **Returns**: `torch.Tensor` (real-valued)

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

Short-time Fourier Transform. Matches `torch.stft` signature.

- **Returns**: `ComplexTensor` with shape `(..., freq_bins, num_frames)`

### `istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=True, length=None)`

Inverse STFT via overlap-add reconstruction.

- **length**: Exact output length (truncates or pads).
- **Returns**: `torch.Tensor` (real-valued)

## Algorithms

- **Power-of-2 sizes**: Cooley-Tukey radix-2 (iterative, decimation-in-time)
- **Arbitrary sizes**: Bluestein's chirp-z transform (pads to power-of-2, 3 FFTs)
- **Plan caching**: Plans are cached by `(size, inverse)` — first call computes, subsequent calls reuse.
