"""Feature-block contract: every feature declares its availability delay.

Anti-leakage core of the feature engineering. A feature column with delay
``d`` (hours) at timestamp ``t`` may only use information known at or before
``t - d``. At forecast origin ``as_of`` the feature is usable at ``t`` iff
``t - d <= as_of``. In a day-ahead backtest (horizon H = 24 h) every
load-derived feature therefore needs ``d >= H`` to be usable on the whole
test block [as_of + 1, as_of + H].
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

    ``delays`` maps every column of ``frame`` to its availability delay in
    hours: column ``c`` at time ``t`` is known at forecast origin ``as_of``
    iff ``t - delays[c] <= as_of``. Deterministic features (calendar) and
    archived forecasts (weather) have delay 0; load-derived features declare
    their lag (>= horizon for day-ahead safety).
    """

    name: str
    frame: pd.DataFrame
    delays: dict[str, int]

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

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)

    def available(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Boolean mask (index x columns), True where known at ``as_of``."""
        idx = self.frame.index
        return pd.DataFrame(
            {c: (idx - pd.Timedelta(hours=d)) <= as_of for c, d in self.delays.items()},
            index=idx,
        )

    def apply_as_of(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Frame with NaN wherever a feature is not yet known at ``as_of``."""
        return self.frame.where(self.available(as_of))
