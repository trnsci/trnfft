"""Host results through the caching host allocator (trnfft#94 follow-up, as trnblas#61/#64).

CPU-only: there is no accelerator to pin for, so `torch.empty(pin_memory=True)`
is recorded and stripped, and `_neuron_host_allocator` is forced. The bit
identity and the speed are measured on hardware (`tests/test_fft.py`,
`TestPinnedOutputNeuron`).
"""

import pytest
import torch

from trnfft import fft_core as fc
from trnfft.complex import ComplexTensor
from trnfft.nki import dispatch
from trnfft.nki import multicore as mc

MiB = 1 << 20


@pytest.fixture
def pins(monkeypatch):
    """Native path, Neuron host allocator; records each pinned allocation's shape."""
    monkeypatch.delenv("TRNFFT_PINNED_OUTPUT", raising=False)
    monkeypatch.setattr(dispatch, "_dispatch_path", "native")
    monkeypatch.setattr(dispatch, "_use_simulator", lambda: False)
    monkeypatch.setattr(dispatch, "_neuron_host_allocator", lambda: True)
    calls = []
    real_empty = torch.empty

    def fake_empty(*args, pin_memory=False, **kw):
        if pin_memory:
            calls.append(tuple(args[0]))
        return real_empty(*args, **kw)

    monkeypatch.setattr(dispatch.torch, "empty", fake_empty)
    return calls


class TestGate:
    def test_use_host_cache(self, pins, monkeypatch):
        assert dispatch._use_host_cache("cpu", MiB)
        assert not dispatch._use_host_cache("cpu", MiB - 1)
        assert not dispatch._use_host_cache("meta", 64 * MiB)
        for value in ("0", "false", "off", "no"):
            monkeypatch.setenv("TRNFFT_PINNED_OUTPUT", value)
            assert not dispatch._use_host_cache("cpu", 64 * MiB)
        monkeypatch.setenv("TRNFFT_PINNED_OUTPUT", "1")
        assert dispatch._use_host_cache("cpu", 64 * MiB)
        monkeypatch.setattr(dispatch.torch.compiler, "is_compiling", lambda: True)
        assert not dispatch._use_host_cache("cpu", 64 * MiB)

    def test_needs_the_neuron_host_allocator(self, pins, monkeypatch):
        # A CUDA build with no GPU raises on pin_memory=True: never pin there.
        monkeypatch.setattr(dispatch, "_neuron_host_allocator", lambda: False)
        assert not dispatch._use_host_cache("cpu", 64 * MiB)
        dispatch._host_empty((512, 1024), torch.float32)
        assert pins == []

    def test_real_allocator_probe_on_this_host(self):
        # No Neuron accelerator in the CPU job: the probe must say no, not raise.
        assert dispatch._neuron_host_allocator() is False

    def test_off_the_native_path(self, pins, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "xla")
        assert not dispatch._use_host_cache("cpu", 64 * MiB)
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_use_simulator", lambda: True)
        assert not dispatch._use_host_cache("cpu", 64 * MiB)


class TestHostEmpty:
    def test_large_is_pinned(self, pins):
        t = dispatch._host_empty((512, 512), torch.float32)
        assert pins == [(512, 512)] and t.shape == (512, 512) and t.dtype == torch.float32

    def test_small_stays_pageable(self, pins):
        dispatch._host_empty((511, 512), torch.float32)
        assert pins == []

    def test_non_cpu_destination(self, pins):
        t = dispatch._host_empty((512, 1024), torch.float32, device="meta")
        assert pins == [] and t.device.type == "meta"

    def test_results_never_alias(self, pins):
        a = dispatch._host_empty((512, 1024), torch.float32)
        b = dispatch._host_empty((512, 1024), torch.float32)
        assert a.data_ptr() != b.data_ptr()

    def test_to_host_leaves_host_tensors(self, pins):
        t = torch.randn(512, 1024)
        assert dispatch._to_host(t) is t  # already on the host: no copy
        assert pins == []


class TestDriverExits:
    """Every host-output exit of the FFT drivers goes through `_to_host`."""

    @pytest.fixture
    def exits(self, monkeypatch):
        seen = []

        def rec(t, device="cpu"):
            seen.append((tuple(t.shape), torch.device(device).type))
            return t.to(device)

        monkeypatch.setattr(fc, "_to_host", rec)
        return seen

    def test_six_step(self, exits):
        g = torch.Generator().manual_seed(97)
        xr, xi = torch.randn(3, 1 << 13, generator=g), torch.randn(3, 1 << 13, generator=g)
        y = fc._fft_via_six_step(ComplexTensor(xr, xi), inverse=False)
        assert exits[-2:] == [((3, 1 << 13), "cpu")] * 2
        ref = torch.fft.fft(torch.complex(xr.double(), xi.double()))
        got = torch.complex(y.real.double(), y.imag.double())
        assert float((got - ref).norm() / ref.norm()) < 1e-5


class TestLaunch:
    def test_launch_results_through_to_host(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_dispatch_path", "native")
        monkeypatch.setattr(dispatch, "_NEURON_DEVICE", "cpu")
        monkeypatch.setattr(dispatch, "_native_ops", {"fake": lambda a, b: (a + b, a - b)})
        seen = []

        def rec(t, device="cpu"):
            seen.append(torch.device(device).type)
            return t.to(device)

        monkeypatch.setattr(dispatch, "_to_host", rec)
        a, b = torch.ones(2, 3), torch.ones(2, 3)
        out = dispatch._launch(None, "fake", a, b)
        assert seen == ["cpu", "cpu"] and torch.equal(out[0], a + b)


class TestBatchSplitIntoHostOutput:
    """The native batch split copies each shard's result into its rows of one
    host output instead of concatenating fresh pageable shards."""

    @pytest.mark.parametrize("batch,cores", [(4096, 2), (999, 3)])
    def test_bit_identical(self, pins, monkeypatch, batch, cores):
        monkeypatch.setattr(mc, "HAS_TORCH_NEURONX", True)
        monkeypatch.setattr(mc, "_core_devices", lambda src, k: [torch.device("cpu")] * k)
        mc._WARM.clear()
        g = torch.Generator().manual_seed(batch)
        xr, xi = torch.randn(batch, 128, generator=g), torch.randn(batch, 128, generator=g)
        x = ComplexTensor(xr, xi)
        got = mc._batch_split_fft(x, inverse=False, num_cores=cores)
        want = fc.fft_core(x, inverse=False)
        assert torch.equal(got.real, want.real) and torch.equal(got.imag, want.imag)
        big = batch * 128 * 4 >= MiB
        assert pins == ([(batch, 128)] * 2 if big else [])
        again = mc._batch_split_fft(x, inverse=False, num_cores=cores)  # warm: threads
        assert torch.equal(again.real, want.real) and torch.equal(again.imag, want.imag)
