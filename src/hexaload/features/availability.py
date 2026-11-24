"""Assembly of feature blocks: common index, manifest, as-of masking.

The manifest records, for every feature column, which block produced it and
its availability delay in hours — the machine-readable "blinkers" that make
leakage auditable (OpenSTEF-style: version every lag by its date of
availability).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .base import FeatureBlock

Manifest = dict[str, dict[str, Any]]


def stack_blocks(
    blocks: Iterable[FeatureBlock], index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, Manifest]:
    """Concatenate blocks on the common UTC index; build the manifest."""
    frames: list[pd.DataFrame] = []
    manifest: Manifest = {}
    for block in blocks:
        frame = block.frame
        if not frame.index.equals(index):
            frame = frame.reindex(index)
        frames.append(frame)
        for col, delay in block.delays.items():
            if col in manifest:
                raise ValueError(f"duplicate feature column across blocks: {col}")
            manifest[col] = {"block": block.name, "delay_h": int(delay)}
    if not frames:
        return pd.DataFrame(index=index), manifest
    return pd.concat(frames, axis=1), manifest


def apply_as_of(
    X: pd.DataFrame, manifest: Manifest, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Mask ``X``: NaN at ``t`` for every column whose delay makes it unknown
    at forecast origin ``as_of`` (i.e. ``t - delay > as_of``)."""
    out = X.copy()
    idx = X.index
    for col, meta in manifest.items():
        delay = int(meta["delay_h"])
        if delay == 0:
            out[col] = out[col].where(idx <= as_of)
        else:
            out[col] = out[col].where((idx - pd.Timedelta(hours=delay)) <= as_of)
    return out


def assert_horizon_safe(
    manifest: Manifest, *, horizon: int, load_blocks: Iterable[str] = ("lags",)
) -> None:
    """Load-derived blocks must declare delay >= ``horizon`` to be usable on
    a whole test block [as_of+1, as_of+horizon]; refuse otherwise.

    Deterministic (calendar) and archived-forecast (weather) blocks have
    delay 0 and are always safe: they never read the target series.
    """
    load = set(load_blocks)
    bad = sorted(
        col
        for col, meta in manifest.items()
        if meta["block"] in load and int(meta["delay_h"]) < horizon
    )
    if bad:
        raise ValueError(
            f"leakage risk: load-derived features with delay < horizon "
            f"{horizon}h: {bad}"
        )
