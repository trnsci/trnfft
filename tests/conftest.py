"""Test configuration and fixtures."""

import atexit
import os
import shutil
import tempfile

import pytest

# NKI trace cache (aws-neuron/aws-neuron-sdk#1454, trnsci/trnsci#35).
# `import torch_neuronx` turns the cache on (NKI_ENABLE_TRACE_CACHE=1), so
# that is how users run these kernels, and the neuron tests run the same way.
# The cache lives on disk and is versioned by wheel .dist-info, which an
# editable install or source checkout does not change, so a shared cache could
# serve a trace of an older kernel. Each session therefore gets a fresh,
# empty cache directory (NKI_TRACE_CACHE_URL), removed at exit.
# tests/test_trace_cache_safety.py checks that no kernel depends on a value
# the cache key misses (closure or global). Overrides: NKI_ENABLE_TRACE_CACHE=0
# runs with the cache off; an explicit NKI_TRACE_CACHE_URL is kept as is.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "1")
if os.environ["NKI_ENABLE_TRACE_CACHE"] != "0" and "NKI_TRACE_CACHE_URL" not in os.environ:
    _trace_cache_dir = tempfile.mkdtemp(prefix="trnfft-nki-trace-cache-")
    os.environ["NKI_TRACE_CACHE_URL"] = _trace_cache_dir
    atexit.register(shutil.rmtree, _trace_cache_dir, ignore_errors=True)
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
