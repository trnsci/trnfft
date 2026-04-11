"""Test configuration and fixtures."""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: requires Neuron hardware")


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
def random_complex_signal(rng):
    def _make(n, batch=None):
        from trnfft import ComplexTensor
        shape = (batch, n) if batch else (n,)
        return ComplexTensor(
            torch.randn(shape, generator=rng),
            torch.randn(shape, generator=rng),
        )
    return _make
