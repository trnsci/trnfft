"""Double-word FFT on Trainium (trnfft#172), native TorchNeuron path.

Checks for the hardware window (the device checklist on #172):

1. bitwise: the kernel equals the CPU reference ``dw.reference_fft`` bit for
   bit at every size 2..4096, both addition kinds, batch tails included
   (the E0 probe of trnsci/trnsci#169 established RN fp32 on VectorE and
   GpSimd, which this relies on);
2. accuracy: correct bits against mpmath on device results;
3. LNC: run this file at NEURON_LOGICAL_NC_CONFIG=1 and =2; the
   ``test_lnc_digest`` line printed by both runs must match (and equals the
   CPU digest, which the test asserts);
4. the public API on device-resident tensors, and DW chaining on device.

Timing is not asserted here; ``scripts/bench_dw_fft.py`` measures it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from trnfft import dw
from trnfft.nki import dw_fft as K
from trnfft.nki.dispatch import HAS_NKI, _get_dispatch_path, _use_simulator


def _native() -> bool:
    return HAS_NKI and not _use_simulator() and _get_dispatch_path() == "native"


pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(not _native(), reason="requires TorchNeuron (native) dispatch path"),
]

SIZES = [1 << m for m in range(1, 13)]


def _input(B, n, seed, spread=20):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, n)) + 1j * rng.standard_normal((B, n))
    if spread:  # rng.integers(0, 0) raises (high <= 0)
        x *= 2.0 ** rng.integers(-spread, spread, (B, 1))
    return dw.to_dw(torch.from_numpy(x))


def _host(y: dw.DWComplex) -> dw.DWComplex:
    return dw.DWComplex(*(t.to("cpu") for t in y))


def _bitwise(a, b) -> bool:
    return all(
        torch.equal(p.contiguous().view(torch.int32), q.contiguous().view(torch.int32))
        for p, q in zip(a, b, strict=True)
    )


def _digest(y: dw.DWComplex) -> str:
    h = hashlib.sha256()
    for t in y:
        h.update(t.contiguous().view(torch.int32).numpy().tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("add", dw.ADD_MODES)
@pytest.mark.parametrize("n", SIZES)
def test_kernel_equals_reference_bitwise(n, add):
    x = _input(128, n, seed=n)
    got = _host(K.run(x, add))
    assert _bitwise(got, dw.reference_fft(x, add))


@pytest.mark.parametrize("n,B", [(4, 1), (16, 300), (512, 130), (2048, 129), (4096, 3), (64, 4096)])
def test_kernel_batch_tails_bitwise(n, B):
    x = _input(B, n, seed=B)
    assert _bitwise(_host(K.run(x, "accurate")), dw.reference_fft(x))


def test_lnc_digest(capsys):
    """Digest of a fixed set of transforms; equal to the CPU reference, and
    printed so the LNC=1 and LNC=2 runs can be compared."""
    parts = []
    for n in (8, 64, 512, 4096):
        x = _input(256, n, seed=7 * n)
        got = _host(K.run(x, "accurate"))
        assert _bitwise(got, dw.reference_fft(x))
        parts.append(_digest(got))
    with capsys.disabled():
        print(f"\ntest_lnc_digest LNC={K.lnc()} {' '.join(parts)}")


@pytest.mark.parametrize("n", [4, 64, 1024, 4096])
def test_device_accuracy_vs_mpmath(n):
    from test_dw import _FLOOR, _bits, _mp_dft  # tests/ is on sys.path (rootdir conftest)

    x = _input(2, n, seed=100 + n, spread=0)
    xe = x.to_complex128().numpy()
    y = _host(K.run(x, "accurate")).to_complex128().numpy()
    for i in range(2):
        assert _bits(y[i], _mp_dft(xe[i])) >= _FLOOR[n]


def test_public_api_device_tensors():
    import trnfft
    from trnfft.complex import ComplexTensor
    from trnfft.fft_core import fft_core

    x = torch.randn(8, 1024, generator=torch.Generator().manual_seed(3))
    xd = x.to("neuron")
    y = fft_core(ComplexTensor(xd, torch.zeros_like(xd)), precision="dw")
    assert y.real.device.type == "neuron" and y.real.dtype == torch.float32
    ref = dw.fft_dw(ComplexTensor(x, torch.zeros_like(x)), backend="reference")
    assert torch.equal(y.real.cpu(), ref.real.float()) and torch.equal(
        y.imag.cpu(), ref.imag.float()
    )
    prev = trnfft.get_backend()
    trnfft.set_backend("nki")
    try:
        z = torch.randn(4, 256, dtype=torch.complex128)
        assert torch.equal(dw.fft_dw(z), dw.fft_dw(z, backend="reference"))
        assert torch.equal(
            dw.fft_dw(z, inverse=True), dw.fft_dw(z, inverse=True, backend="reference")
        )
    finally:
        trnfft.set_backend(prev)


def test_dw_chain_stays_on_device():
    """forward then inverse on device-resident DW planes: no host round trip,
    same bits as the reference chain, and back to the input within ~2^-46."""
    x = _input(64, 2048, seed=5)
    xd = x.to("neuron")
    y = dw.fft_dw_raw(xd)
    assert y.device.type == "neuron"
    z = dw.fft_dw_raw(y, inverse=True)
    ref = dw.fft_dw_raw(dw.fft_dw_raw(x, backend="reference"), inverse=True, backend="reference")
    assert _bitwise(_host(z), ref)
    back = _host(z).to_complex128() / 2048
    assert (back - x.to_complex128()).abs().max() <= 2.0**-44 * x.to_complex128().abs().max()
