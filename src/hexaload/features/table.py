"""Assemble feature blocks into one table plus its availability manifest."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .availability import Manifest, apply_as_of, stack_blocks
from .base import as_utc_hourly_index
from .calendar import SCHOOL_VACATIONS_CSV, calendar_block
from .cyclic import cyclic_block

DEFAULT_BLOCKS = ("calendar", "cyclic")


def build_feature_table(
    idx: pd.DatetimeIndex,
    *,
    as_of: pd.Timestamp | None = None,
    blocks: Iterable[str] = DEFAULT_BLOCKS,
    school_csv: Path | None = None,
) -> tuple[pd.DataFrame, Manifest]:
    """Build the feature table over the UTC hourly ``idx``.

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
        else:
            raise ValueError(f"unknown feature block {name!r}")
    X, manifest = stack_blocks(built, idx)
    if as_of is not None:
        X = apply_as_of(X, manifest, as_of)
    return X, manifest
