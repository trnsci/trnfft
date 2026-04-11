# Quickstart

## FFT operations

```python
import torch
import trnfft

# 1D FFT (drop-in replacement for torch.fft.fft)
signal = torch.randn(1024)
X = trnfft.fft(signal)
recovered = trnfft.ifft(X)

# Real-valued FFT (positive frequencies only)
X = trnfft.rfft(signal)
recovered = trnfft.irfft(X, n=1024)

# 2D FFT
image = torch.randn(256, 256)
F = trnfft.fft2(image)

# N-D FFT
volume = torch.randn(8, 16, 32)
F = trnfft.fftn(volume)
recovered = trnfft.ifftn(F)

# STFT / ISTFT
waveform = torch.randn(16000)
S = trnfft.stft(waveform, n_fft=512, hop_length=256)
recovered = trnfft.istft(S, n_fft=512, hop_length=256, length=16000)
```

## Complex tensors

```python
from trnfft import ComplexTensor

# Create from real and imaginary parts
z = ComplexTensor(torch.randn(4, 4), torch.randn(4, 4))

# Arithmetic
w = z * z          # element-wise complex multiply
m = z @ z          # complex matrix multiply
c = z.conj()       # conjugate
mag = z.abs()      # magnitude
phase = z.angle()  # phase
```

## Complex neural network layers

```python
from trnfft import ComplexTensor
from trnfft.nn import ComplexLinear, ComplexConv1d, ComplexModReLU

x = ComplexTensor(torch.randn(8, 256), torch.randn(8, 256))

layer = ComplexLinear(256, 128)
y = layer(x)  # ComplexTensor output

act = ComplexModReLU(128)
y = act(y)  # ReLU on magnitude, preserves phase
```

## Backend selection

```python
import trnfft

# Check if NKI is available
print(trnfft.HAS_NKI)

# Force backend
trnfft.set_backend("pytorch")  # Always use PyTorch ops
trnfft.set_backend("nki")      # Always use NKI (fails if not on Trainium)
trnfft.set_backend("auto")     # NKI if available, else PyTorch (default)
```
