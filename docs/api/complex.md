# ComplexTensor

Trainium has no native complex dtype. `ComplexTensor` stores complex values as paired real tensors and implements complex arithmetic via real-valued operations.

## Construction

```python
from trnfft import ComplexTensor

# From real tensor (imaginary part is zero)
z = ComplexTensor(torch.randn(4, 4))

# From real and imaginary parts
z = ComplexTensor(real_part, imag_part)

# From torch complex tensor
z = ComplexTensor(torch.complex(real, imag))

# From polar coordinates
z = ComplexTensor.from_polar(magnitude, phase)
```

## Properties

- `z.real` — real part (`torch.Tensor`)
- `z.imag` — imaginary part (`torch.Tensor`)
- `z.shape` — tensor shape
- `z.dtype` — element dtype (of the real/imag tensors)
- `z.device` — device

## Arithmetic

| Operation | Syntax |
|-----------|--------|
| Addition | `a + b` |
| Subtraction | `a - b` |
| Element-wise multiply | `a * b` |
| Matrix multiply | `a @ b` |
| Negation | `-a` |
| Scalar multiply | `a * 2.0` |

All operators support `ComplexTensor`, scalar, and real `torch.Tensor` operands.

## Methods

- `z.abs()` — magnitude (`torch.Tensor`)
- `z.angle()` — phase (`torch.Tensor`)
- `z.conj()` — complex conjugate
- `z.to_torch_complex()` — convert to `torch.complex64/128`
- `z.clone()` — deep copy
- `z.to(device)` — move to device
- `z.reshape(*shape)` — reshape
- `z.transpose(dim0, dim1)` — transpose
- `z.unsqueeze(dim)` / `z.squeeze(dim)` — dimension manipulation
- `z[key]` — indexing/slicing (returns `ComplexTensor`)

## `complex_matmul(a, b)`

Complex matrix multiply decomposed into 4 real matmuls:

```
C_real = A_real @ B_real - A_imag @ B_imag
C_imag = A_real @ B_imag + A_imag @ B_real
```
