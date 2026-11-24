"""Assemble feature blocks into one table plus its availability manifest."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .availability import Manifest, apply_as_of, assert_horizon_safe, stack_blocks
from .base import as_utc_hourly_index
from .calendar import SCHOOL_VACATIONS_CSV, calendar_block
from .cyclic import cyclic_block
from .lags import lags_block
from .weather import weather_block

DEFAULT_BLOCKS = ("calendar", "cyclic", "weather", "lags")
DEFAULT_HORIZON = 24  # day-ahead


def build_feature_table(
    idx: pd.DatetimeIndex,
    *,
    as_of: pd.Timestamp | None = None,
    blocks: Iterable[str] = DEFAULT_BLOCKS,
    school_csv: Path | None = None,
    wx: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    prevision_j1: pd.Series | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, Manifest]:
    """Build the feature table over the UTC hourly ``idx``.

    Blocks: ``calendar`` / ``cyclic`` (deterministic, delay 0), ``weather``
    (archived forecasts, delay 0; needs ``wx``), ``lags`` (load memory; needs
    ``y``, optional ``prevision_j1``; delays >= ``horizon``, asserted).

    With ``as_of`` (forecast origin), every column is masked to NaN at ``t``
    when ``t - delay > as_of`` — the structural anti-leakage guarantee. The
    returned manifest maps each column to ``{"block", "delay_h"}``.
    """
    idx = as_utc_hourly_index(idx)
    built = []
    for name in blocks:
        if name == "calendar":
            built.append(
                calendar_block(idx, school_csv=school_csv or SCHOOL_VACATIONS_CSV)
            )
        elif name == "cyclic":
            built.append(cyclic_block(idx))
        elif name == "weather":
            if wx is None:
                raise ValueError("block 'weather' requires wx=<national weather frame>")
            built.append(weather_block(wx, index=idx))
        elif name == "lags":
            if y is None:
                raise ValueError("block 'lags' requires y=<load series>")
            built.append(
                lags_block(y, index=idx, prevision_j1=prevision_j1, horizon=horizon)
            )
        else:
            raise ValueError(f"unknown feature block {name!r}")
    X, manifest = stack_blocks(built, idx)
    assert_horizon_safe(manifest, horizon=horizon)
    if as_of is not None:
        X = apply_as_of(X, manifest, as_of)
    return X, manifest
