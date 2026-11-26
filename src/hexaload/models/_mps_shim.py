"""MPS float64 shim: let float64-casting code run on Apple Silicon (MPS).

PyTorch's MPS backend has no float64 kernels: ``.to(torch.float64)`` on an MPS
tensor raises ``TypeError``. Timestral's weights are float32, but its causal
normalization scaler accumulates Welford statistics in float64 on the model's
device (``timestral/scaler.py``) and ``_refine_block`` casts the block median
to float64. This shim redirects float64 casts to float32 **only when the tensor
is on MPS or the target device is MPS**; CPU tensors are untouched, so the CPU
reference path stays bit-exact.

Validated in ``notebooks/timestral_lora_mps.ipynb``: max |CPU - MPS| over all
quantiles = 0.035 MW on ~1e4 MW values (pure float32 rounding).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

_APPLIED = False
_ORIG_TO: Callable | None = None
_ORIG_DOUBLE: Callable | None = None


def needs_f64_downcast(
    self_device: str, dtype: torch.dtype | None, target_device: str | None
) -> bool:
    """Pure predicate: does this cast request float64 onto an MPS device?"""
    if dtype is not torch.float64:
        return False
    return self_device == "mps" or target_device == "mps"


def _parse_cast(args: tuple, kwargs: dict) -> tuple[torch.dtype | None, str | None]:
    """Extract ``(dtype, device.type)`` from a ``Tensor.to(*args, **kwargs)`` call."""
    dtype = kwargs.get("dtype")
    dev = kwargs.get("device")
    for a in args:
        if isinstance(a, torch.Tensor):
            if dtype is None:
                dtype = a.dtype
            if dev is None:
                dev = a.device.type
        elif isinstance(a, torch.dtype):
            dtype = a
        elif isinstance(a, (str, torch.device)):
            dev = torch.device(a).type
    if isinstance(dev, torch.device):
        dev = dev.type
    return dtype, dev


def apply_mps_f64_shim() -> None:
    """Install the shim process-wide. Idempotent; no-op when MPS is absent."""
    global _APPLIED, _ORIG_TO, _ORIG_DOUBLE
    if _APPLIED or not torch.backends.mps.is_available():
        return
    _ORIG_TO = torch.Tensor.to
    _ORIG_DOUBLE = torch.Tensor.double

    def safe_to(self: torch.Tensor, *args, **kwargs):
        dtype, dev = _parse_cast(args, kwargs)
        if needs_f64_downcast(self.device.type, dtype, dev):
            args = tuple(torch.float32 if a is torch.float64 else a for a in args)
            if kwargs.get("dtype") is torch.float64:
                kwargs = dict(kwargs, dtype=torch.float32)
        return _ORIG_TO(self, *args, **kwargs)

    def safe_double(self: torch.Tensor):
        if needs_f64_downcast(self.device.type, torch.float64, None):
            return _ORIG_TO(self, torch.float32)
        return _ORIG_DOUBLE(self)

    torch.Tensor.to = safe_to
    torch.Tensor.double = safe_double
    _APPLIED = True


@contextmanager
def mps_f64_shim() -> Iterator[None]:
    """Context-manager form (tests): install the shim, restore afterwards."""
    global _APPLIED, _ORIG_TO, _ORIG_DOUBLE
    if not torch.backends.mps.is_available():
        yield
        return
    saved = (_APPLIED, _ORIG_TO, _ORIG_DOUBLE, torch.Tensor.to, torch.Tensor.double)
    apply_mps_f64_shim()
    try:
        yield
    finally:
        _APPLIED, _ORIG_TO, _ORIG_DOUBLE = saved[0], saved[1], saved[2]
        torch.Tensor.to = saved[3]
        torch.Tensor.double = saved[4]
