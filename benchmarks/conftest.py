"""Benchmark fixtures."""

import pytest
import torch


@pytest.fixture(params=[256, 1024, 4096, 16384, 65536])
def fft_size(request):
    return request.param


@pytest.fixture(params=[128, 256, 512, 1024])
def gemm_size(request):
    return request.param


@pytest.fixture
def random_signal(fft_size):
    torch.manual_seed(42)
    return torch.randn(fft_size)


@pytest.fixture
def random_complex_matrix(gemm_size):
    torch.manual_seed(42)
    from trnfft import ComplexTensor

    return ComplexTensor(
        torch.randn(gemm_size, gemm_size),
        torch.randn(gemm_size, gemm_size),
    )
