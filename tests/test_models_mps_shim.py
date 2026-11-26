"""MPS float64 shim — pure logic tests, no real GPU probing (CI-safe).

Like test_models_device.py: MPS availability is monkeypatched and no MPS
tensor is ever allocated (touching MPS under pytest's threads can segfault).
"""

import pytest
import torch

from hexaload.models._mps_shim import (
    _parse_cast,
    apply_mps_f64_shim,
    mps_f64_shim,
    needs_f64_downcast,
)


def test_predicate() -> None:
    assert needs_f64_downcast("mps", torch.float64, None)
    assert needs_f64_downcast("cpu", torch.float64, "mps")  # .to("mps", torch.float64)
    assert not needs_f64_downcast("cpu", torch.float64, None)
    assert not needs_f64_downcast("mps", torch.float32, None)
    assert not needs_f64_downcast("mps", None, None)


def test_parse_cast() -> None:
    assert _parse_cast((torch.float64,), {}) == (torch.float64, None)
    assert _parse_cast(("mps", torch.float64), {}) == (torch.float64, "mps")
    assert _parse_cast((), {"device": "mps", "dtype": torch.float64}) == (
        torch.float64,
        "mps",
    )
    t = torch.ones(2)
    assert _parse_cast((t,), {}) == (torch.float32, "cpu")
    assert _parse_cast((torch.device("cpu"),), {}) == (None, "cpu")


def test_noop_without_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    orig_to = torch.Tensor.to
    apply_mps_f64_shim()
    assert torch.Tensor.to is orig_to  # untouched


def test_cpu_path_bit_exact_under_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with mps_f64_shim():
        x = torch.ones(4)  # CPU tensor: the shim must not touch it
        assert x.to(torch.float64).dtype == torch.float64
        assert x.double().dtype == torch.float64
        assert x.to("cpu", torch.float64).dtype == torch.float64


def test_apply_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with mps_f64_shim():
        apply_mps_f64_shim()
        first = torch.Tensor.to
        apply_mps_f64_shim()
        assert torch.Tensor.to is first
