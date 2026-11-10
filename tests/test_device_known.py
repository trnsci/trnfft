"""CPU checks for the device-known conformance marking (trnsci/trnsci#35)."""

from types import SimpleNamespace

import conftest
import pytest
import test_conformance as tc


def _node(name, **params):
    return SimpleNamespace(name=name, originalname=name, callspec=SimpleNamespace(params=params))


@pytest.mark.parametrize(
    "node,cat",
    [
        (
            _node("test_complex_conv1d", kw=dict(stride=3, padding=2), length=1),
            "conv1d-len1-stride-pad",
        ),
        (_node("test_complex_conv1d", kw=dict(stride=3, padding=2), length=10), None),
        (_node("test_complex_conv1d", kw=dict(stride=2), length=1), None),
        (_node("test_complex_conv1d", kw=dict(padding=1), length=1), None),
        (_node("test_complex_linear", shape=(6,), bias=True), None),
        (SimpleNamespace(name="test_plain", originalname="test_plain"), None),
    ],
)
def test_device_known_category(node, cat):
    assert tc._device_known_category(node) == cat
    if cat is not None:
        assert cat in tc.DEVICE_KNOWN


def test_every_category_names_an_issue():
    for issue, what in tc.DEVICE_KNOWN.values():
        assert issue.startswith("trnsci/trnfft#") and what


def test_summary_categories():
    assert conftest._expect_category("[device-known:conv1d-len1-stride-pad] x: y") == (
        "conv1d-len1-stride-pad"
    )
    assert conftest._expect_category("trnsci/trnfft#106: window") == "other xfail"
    assert conftest._expect_category(None) == "other xfail"

    def rep(nodeid, wasxfail=None):
        r = SimpleNamespace(nodeid=nodeid, when="call")
        if wasxfail is not None:
            r.wasxfail = wasxfail
        return r

    s = conftest._expect_summary(
        {
            "passed": [rep("a"), rep("b")],
            "xfailed": [rep("c", "[device-known:k1] i: w")],
            "xpassed": [rep("d", "[device-known:k2] i: w")],
            "failed": [rep("e")],
        }
    )
    assert s["passed"] == 2
    assert s["xfailed"] == {"k1": ["c"]}
    assert s["xpassed"] == {"k2": ["d"]}
    assert s["new_failures"] == ["e"]
