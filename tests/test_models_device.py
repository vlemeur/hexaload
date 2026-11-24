"""Torch device resolution — pure logic tests.

Never probe the real GPU from the test suite: touching MPS under pytest's
threads can segfault on macOS. All GPU probes are monkeypatched.
"""

import pytest
import torch

from hexaload.models._device import default_device, resolve_device


def _no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_auto_cpu_when_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch)
    assert default_device() == "cpu"


def test_auto_mps_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == "mps"


def test_cuda_has_priority_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == "cuda"


def test_override_cpu_and_allocation_on_cpu() -> None:
    assert resolve_device("cpu") == "cpu"
    x = torch.zeros(2, device=resolve_device("cpu"))  # CPU allocation is safe
    assert x.device.type == "cpu"
    assert float(x.sum()) == 0.0


def test_mps_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch)
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps") == "mps"


def test_unknown_device_raises() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")
