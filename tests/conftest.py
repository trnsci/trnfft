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
    # tests/test_conformance.py: the same reference cases on any backend/device.
    # CPU CI uses the defaults; a Trn2 run passes e.g.
    # `--backend nki --conformance-device neuron` (trnsci/trnsci#105).
    parser.addoption(
        "--backend",
        action="store",
        default="pytorch",
        choices=("pytorch", "nki", "auto"),
        help="trnfft backend for the conformance suite (default: pytorch)",
    )
    parser.addoption(
        "--conformance-device",
        action="store",
        default="cpu",
        help="device for conformance-suite inputs, e.g. cpu or neuron (default: cpu)",
    )
    parser.addoption(
        "--conformance-json",
        action="store",
        default=None,
        metavar="PATH",
        help="write the device conformance summary (passed, xfailed/XPASS by category, "
        "NEW failures) as JSON",
    )


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


# ---------------------------------------------------------------- device conformance summary
# Device conformance runs (``--backend nki --conformance-device neuron``) mark
# known platform limits ``xfail(strict=False)`` at run time, with a reason that
# starts ``[device-known:<category>]`` (see DEVICE_KNOWN in
# tests/test_conformance.py). On a non-CPU device run this prints passed,
# xfailed and XPASS by category, and the NEW failures (anything not expected).
# ``--conformance-json PATH`` also writes it as JSON; merge the JSON from
# chunked runs with ``python scripts/conformance_expect.py run1.json ...``.

_KNOWN_PREFIX = "[device-known:"


def _expect_category(reason):
    reason = reason or ""
    if reason.startswith(_KNOWN_PREFIX):
        return reason[len(_KNOWN_PREFIX) :].split("]", 1)[0]
    return "other xfail"


def _expect_summary(stats):
    def ids(key):
        return [
            r.nodeid for r in stats.get(key, []) if getattr(r, "when", "call") in ("call", "setup")
        ]

    by_cat = {}
    for key in ("xfailed", "xpassed"):
        for r in stats.get(key, []):
            cat = _expect_category(getattr(r, "wasxfail", ""))
            by_cat.setdefault(key, {}).setdefault(cat, []).append(r.nodeid)
    return {
        "passed": len(ids("passed")),
        "skipped": len(stats.get("skipped", [])),
        "xfailed": by_cat.get("xfailed", {}),
        "xpassed": by_cat.get("xpassed", {}),
        "new_failures": ids("failed") + [r.nodeid for r in stats.get("error", [])],
    }


def _print_expect_summary(write, s):
    write("conformance expectations (device run)")
    write(f"  passed: {s['passed']}   skipped: {s['skipped']}")
    for key, label in (("xfailed", "xfailed"), ("xpassed", "XPASS (known category now passes)")):
        total = sum(len(v) for v in s[key].values())
        write(f"  {label}: {total}")
        for cat, nodeids in sorted(s[key].items()):
            write(f"    {cat}: {len(nodeids)}")
    write(f"  NEW failures: {len(s['new_failures'])}")
    for nodeid in s["new_failures"]:
        write(f"    {nodeid}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    path = config.getoption("--conformance-json", None)
    if config.getoption("--conformance-device", "cpu") == "cpu" and not path:
        return
    s = _expect_summary(terminalreporter.stats)
    _print_expect_summary(lambda line: terminalreporter.write_line(line), s)
    if path:
        import json

        with open(path, "w") as fh:
            json.dump(s, fh, indent=1)
