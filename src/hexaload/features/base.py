"""Feature-block contract: every feature declares its availability delay.

Anti-leakage core of the feature engineering. A target-reading feature column
with delay ``d`` (hours) at timestamp ``t`` may only use information known at
or before ``t - d``; at forecast origin ``as_of`` it is usable at ``t`` iff
``t - d <= as_of``. In a day-ahead backtest (horizon H = 24 h) every
load-derived feature therefore needs ``d >= H`` on the whole test block.

Exogenous features (calendar, archived weather forecasts) do not read the
target: ``reads_target=False``, all delays 0, and they are known in advance —
they stay available on the whole horizon, future included.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def as_utc_hourly_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Coerce a tz-aware DatetimeIndex to UTC (the storage timezone)."""
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("expected a pd.DatetimeIndex")
    if idx.tz is None:
        raise ValueError("index must be timezone-aware; store UTC everywhere")
    return idx.tz_convert("UTC")


@dataclass(frozen=True)
class FeatureBlock:
    """A named set of feature columns on a shared UTC hourly index.

    ``delays`` maps every column of ``frame`` to its lag on the TARGET series
    in hours. ``reads_target`` marks blocks derived from the target (lags):
    column ``c`` at time ``t`` is known at forecast origin ``as_of`` iff
    ``t - delays[c] <= as_of``, and is masked to NaN beyond that. Exogenous
    blocks (``reads_target=False``: calendar, archived weather forecasts) are
    known in advance and stay available on the whole horizon; their delays
    must all be 0.
    """

    name: str
    frame: pd.DataFrame
    delays: dict[str, int]
    reads_target: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise TypeError(f"{self.name}: frame index must be a DatetimeIndex")
        if self.frame.index.tz is None:
            raise ValueError(f"{self.name}: frame index must be timezone-aware (UTC)")
        missing = sorted(set(self.frame.columns) - set(self.delays))
        extra = sorted(set(self.delays) - set(self.frame.columns))
        if missing or extra:
            raise ValueError(
                f"{self.name}: delays/columns mismatch "
                f"(missing={missing}, extra={extra})"
            )
        if any(d < 0 for d in self.delays.values()):
            raise ValueError(f"{self.name}: delays must be >= 0 hours")
        # exogenous blocks may carry delay > 0 (e.g. archived-forecast lags read
        # the run issued at t-k): apply_as_of masks them uniformly, so they are
        # still leakage-safe — only their reads_target=False nature matters for
        # exogenous_columns().

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)

    def available(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Boolean mask (index x columns), True where known at ``as_of``.

        Only meaningful for target-reading blocks; exogenous blocks are
        always available.
        """
        idx = self.frame.index
        return pd.DataFrame(
            {c: (idx - pd.Timedelta(hours=d)) <= as_of for c, d in self.delays.items()},
            index=idx,
        )

    def apply_as_of(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Mask target-reading features to NaN where unknown at ``as_of``.

        Exogenous blocks (reads_target=False) are returned unchanged: their
        inputs (calendar, archived forecasts) are known in advance.
        """
        if not self.reads_target:
            return self.frame.copy()
        return self.frame.where(self.available(as_of))
