"""Double-word FFT kernel on CPU (trnfft#172): MLIR verification at trn2
LNC=1 and LNC=2 (``source_to_mlir(...).verify()``, which ``nki.simulate``
does not run), and ``nki.simulate`` runs that must equal the CPU reference
``dw.reference_fft`` bit for bit at LNC=1 and LNC=2, over sizes 2..4096 and
batch tails. Needs the ``nki`` wheel (Linux x86_64); no device, no
neuronx-cc."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nki_simulator

from trnfft import dw  # noqa: E402
from trnfft.nki import dw_fft as K  # noqa: E402

try:
    from nki.compiler.frontend import FrontendKind, source_to_mlir
except ImportError:
    source_to_mlir = None


@pytest.fixture(autouse=True)
def _nki_available():
    if not K.HAS_NKI or source_to_mlir is None:
        pytest.skip("nki (with nki.compiler.frontend) not importable on this host")


def _dw_input(B, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, n)) + 1j * rng.standard_normal((B, n))
    x *= 2.0 ** rng.integers(-20, 20, (B, 1))
    return dw.to_dw(torch.from_numpy(x))


def _bitwise(a: dw.DWComplex, b: dw.DWComplex) -> bool:
    return all(
        torch.equal(p.contiguous().view(torch.int32), q.contiguous().view(torch.int32))
        for p, q in zip(a, b, strict=True)
    )


# (n, B): every size at a small batch; Rb > 1 rows per partition, several
# tiles, partial partitions and the LNC=2 split at the larger batches.
VERIFY_CASES = [(1 << m, 3) for m in range(1, 13)] + [
    (64, 4096),  # Rb = 32, F = 2048
    (256, 1000),  # Rb = 8, padded
    (1024, 300),  # Rb = 2, NT > 1
    (4096, 256),  # F = 4096, CW = 256, NT = 2
]


KERNELS = sorted(K.KERNEL_NAMES)  # (add, engines)


def test_every_dw_kernel_is_listed():
    names = {k for k in vars(K) if k.startswith("dw_fft_") and k.endswith("_kernel")}
    assert names == set(K.KERNEL_NAMES.values())


@pytest.mark.parametrize("lnc", [1, 2])
@pytest.mark.parametrize("variant", KERNELS, ids="-".join)
@pytest.mark.parametrize("n,B", VERIFY_CASES, ids=lambda v: str(v))
def test_dw_kernel_mlir_verifies(n, B, variant, lnc):
    if variant != ("accurate", "auto") and B != 3 and n != 4096:
        pytest.skip("other variants: every size at B = 3, plus N = 4096 at B = 256")
    planes, _ = K.pack(dw.to_dw(torch.zeros(B, n, dtype=torch.complex128)), lnc)
    source_to_mlir(
        K.kernel(*variant),
        *[p.numpy() for p in planes],
        dw.twiddle_table(n).numpy(),
        target="trn2",
        lnc=lnc,
        frontend=FrontendKind.TRACER,
    ).verify()


# (2048, 129), (2048, 200) and (256, 1030) need more than 128 partition rows:
# 128-row tiles (see dw_fft.layout on why no tile straddles row 128).
SIM_CASES = [(1 << m, 3) for m in range(1, 13)] + [
    (16, 300),
    (512, 130),
    (2048, 129),
    (2048, 200),
    (256, 1030),
]


@pytest.mark.parametrize("lnc", ["1", "2"])
@pytest.mark.parametrize("variant", KERNELS, ids="-".join)
@pytest.mark.parametrize("n,B", SIM_CASES, ids=lambda v: str(v))
def test_dw_kernel_equals_reference_bitwise(n, B, variant, lnc, monkeypatch):
    add, engines = variant
    if variant != ("accurate", "auto") and (B != 3 or n not in (2, 4, 8, 512, 4096)):
        pytest.skip("other variants: a subset of sizes (the butterfly code is shared)")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    x = _dw_input(B, n, seed=n + B)
    got = K.run(x, add, simulate=True, engines=engines)
    assert _bitwise(got, dw.reference_fft(x, add))


def test_public_api_under_simulator(monkeypatch):
    """``fft_dw`` / ``precision="dw"`` route to ``nki.simulate`` under
    TRNFFT_USE_SIMULATOR with backend "nki", with the reference's bits."""
    import trnfft
    from trnfft.complex import ComplexTensor
    from trnfft.fft_core import fft_core
    from trnfft.nki import dispatch as D

    monkeypatch.setattr(D, "_USE_SIMULATOR", True)
    prev = trnfft.get_backend()
    trnfft.set_backend("nki")
    try:
        x = _dw_input(4, 256, seed=1)
        assert dw._backend(x, None) == "simulate"
        z = torch.complex(*(a.double() + b.double() for a, b in ((x[0], x[1]), (x[2], x[3]))))
        y = fft_core(ComplexTensor(z.real.contiguous(), z.imag.contiguous()), precision="dw")
        ref = dw.reference_fft(x).to_complex128()
        assert torch.equal(y.real, ref.real) and torch.equal(y.imag, ref.imag)
        yi = dw.fft_dw(z, inverse=True)
        refi = dw.fft_dw(z, inverse=True, backend="reference")
        assert torch.equal(yi, refi)
    finally:
        trnfft.set_backend(prev)
