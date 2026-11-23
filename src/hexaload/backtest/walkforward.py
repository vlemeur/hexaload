"""Walk-forward CV splits — thin wrapper over sktime splitters.

Anti-leakage contract (enforced by tests):
- every test timestamp is strictly after every train timestamp;
- each test block is exactly ``horizon`` consecutive steps;
- everything stays inside [start, end].

Exogenous variables are not handled here. When used, they must be as-of
forecasts (our Open-Meteo archive is), never observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-11-01"
DEFAULT_HORIZON = 24  # hours ahead


@dataclass(frozen=True)
class Split:
    """One walk-forward fold: train timestamps and the horizon to predict."""

    train: pd.DatetimeIndex
    test: pd.DatetimeIndex

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test[0]


def walk_forward_splits(
    y: pd.Series,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    horizon: int = DEFAULT_HORIZON,
    window_length: int,
    step_length: int,
    expanding: bool = False,
) -> list[Split]:
    """Sliding (default) or expanding walk-forward folds over ``y[start:end]``.

    ``window_length`` and ``step_length`` are in steps of ``y`` (hours here).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if window_length < 1 or step_length < 1:
        raise ValueError("window_length and step_length must be >= 1")
    window = y.loc[start:end]
    if len(window) < window_length + horizon:
        raise ValueError(
            f"not enough data: {len(window)} steps in [{start}, {end}] "
            f"< window_length {window_length} + horizon {horizon}"
        )
    fh = np.arange(1, horizon + 1)
    if expanding:
        splitter = ExpandingWindowSplitter(
            fh=fh, initial_window=window_length, step_length=step_length
        )
    else:
        splitter = SlidingWindowSplitter(
            fh=fh, window_length=window_length, step_length=step_length
        )
    return [
        Split(
            train=pd.DatetimeIndex(window.index[train_pos]),
            test=pd.DatetimeIndex(window.index[test_pos]),
        )
        for train_pos, test_pos in splitter.split(window)
    ]
