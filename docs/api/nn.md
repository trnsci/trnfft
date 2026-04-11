# Complex Neural Network Layers

Complex-valued layers that operate on `ComplexTensor`. Used in speech enhancement, physics-informed neural networks, and other complex-domain workloads.

## `ComplexLinear(in_features, out_features, bias=True)`

Complex-valued linear layer using the decomposition:

```
(W_re + iW_im)(x_re + ix_im) = (W_re·x_re - W_im·x_im) + i(W_re·x_im + W_im·x_re)
```

Uses Kaiming initialization on both weight matrices.

## `ComplexConv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)`

Complex-valued 1D convolution. Same decomposition as `ComplexLinear` but with `nn.Conv1d` internally.

## `ComplexBatchNorm1d(num_features, eps=1e-5)`

Batch normalization for complex tensors. Normalizes real and imaginary parts independently (not the covariance-based variant from Trabelsi et al. 2018). The simpler form works well for cIRM speech enhancement.

## `ComplexModReLU(num_features)`

Modulus ReLU activation:

```
f(z) = ReLU(|z| + b) · z / |z|
```

Applies ReLU to the magnitude while preserving phase. The learnable bias `b` allows the network to learn a magnitude threshold.

## Example: cIRM speech enhancement

```python
from trnfft import ComplexTensor, stft, istft
from trnfft.nn import ComplexLinear, ComplexModReLU

class MaskEstimator(nn.Module):
    def __init__(self, n_fft=512, hidden=256):
        super().__init__()
        freq = n_fft // 2 + 1
        self.net = nn.Sequential(
            ComplexLinear(freq, hidden),
            ComplexModReLU(hidden),
            ComplexLinear(hidden, freq),
        )

    def forward(self, noisy_spec):
        # noisy_spec: (batch, freq, time)
        # Process per-frame
        b, f, t = noisy_spec.shape
        x = ComplexTensor(
            noisy_spec.real.permute(0,2,1).reshape(-1, f),
            noisy_spec.imag.permute(0,2,1).reshape(-1, f),
        )
        mask = self.net(x)
        return ComplexTensor(
            mask.real.reshape(b, t, f).permute(0,2,1),
            mask.imag.reshape(b, t, f).permute(0,2,1),
        )
```
