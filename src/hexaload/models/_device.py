"""Torch device selection: CUDA > MPS (Apple Silicon) > CPU, overridable.

Deep models take ``device=None`` (auto) by default; the test suite forces
``"cpu"`` so CI (no GPU) is deterministic. On a Mac with MPS, ``None`` picks
``"mps"`` automatically.
"""

from __future__ import annotations

import torch


def resolve_device(device: str | None = None) -> str:
    """Resolve a torch device string.

    ``None`` = auto-detect (cuda > mps > cpu). An explicit ``"mps"`` on a
    machine without MPS raises instead of silently falling back.
    """
    if device is not None:
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps is not available")
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"unknown device {device!r} (cpu|cuda|mps)")
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_device() -> str:
    """Auto device for this machine (cuda > mps > cpu)."""
    return resolve_device(None)
