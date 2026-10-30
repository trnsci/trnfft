"""Test configuration and fixtures."""

import os

import pytest

# Test runs come from source checkouts / editable installs, which the NKI trace
# cache cannot version (it keys on wheel .dist-info). A stale hit would run an
# old kernel, so disable it before anything imports nki. Set it explicitly to
# override.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")
import torch


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests marked slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")


def pytest_collection_modifyitems(config, items):
    # Slow tests (e.g. the N=8193 Kahan-Bluestein characterization, ~19 min
    # on Trn2) stay out of the default `-m neuron` run. Opt in with
    # --run-slow, or select them with a marker expression naming "slow".
    if config.getoption("--run-slow") or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow; use --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture(params=[8, 16, 32, 64, 128, 256, 512, 1024])
def power_of_2_size(request):
    return request.param


@pytest.fixture(params=[7, 11, 13, 17, 100, 127, 200, 997])
def arbitrary_size(request):
    return request.param


@pytest.fixture
def random_real_signal(rng):
    def _make(n, batch=None):
        shape = (batch, n) if batch else (n,)
        return torch.randn(shape, generator=rng)

    return _make


@pytest.fixture
def nki_backend():
    """Temporarily set backend to NKI for a test."""
    from trnfft import get_backend, set_backend

    old = get_backend()
    set_backend("nki")
    yield
    set_backend(old)


@pytest.fixture
def random_complex_signal(rng):
    def _make(n, batch=None):
        from trnfft import ComplexTensor

        shape = (batch, n) if batch else (n,)
        return ComplexTensor(
            torch.randn(shape, generator=rng),
            torch.randn(shape, generator=rng),
        )

    return _make
