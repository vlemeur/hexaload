"""Load-memory features (block "lags") — every delay >= horizon.

Day-ahead safe by construction: a feature at ``t`` only reads ``y`` up to
``t - delay`` with ``delay >= horizon`` (H = 24 h), so the whole test block
[as_of+1, as_of+H] is covered. Enforced by ``assert_horizon_safe`` and by the
future-perturbation test.

- direct lags y(t-L), L in (24, 48, 168, 336) -> delay L
- same-week smooth (±1 h around t-168) -> delay 167
- rolling mean/min/max/std over 24 h / 7 d / 30 d, shifted by ``horizon``
  (window ends at t - horizon) -> delay = horizon
- RTE's D-1 forecast prevision_j1(t), published ~24 h ahead -> delay 24
"""

from __future__ import annotations

import pandas as pd

from .base import FeatureBlock, as_utc_hourly_index

LOAD_LAGS_H = (24, 48, 168, 336)
ROLL_WINDOWS_H = (24, 24 * 7, 24 * 30)
ROLL_STATS = ("mean", "min", "max", "std")
RTE_D1_DELAY_H = 24  # prevision_j1(t) is published ~24 h ahead


def lags_block(
    y: pd.Series,
    *,
    index: pd.DatetimeIndex,
    prevision_j1: pd.Series | None = None,
    horizon: int = 24,
) -> FeatureBlock:
    """Load-memory features on the UTC hourly ``index`` (delays >= horizon)."""
    index = as_utc_hourly_index(index)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    y = y.reindex(index)

    frame: dict[str, pd.Series] = {}
    delays: dict[str, int] = {}
    for lag in LOAD_LAGS_H:
        col = f"lag_load_{lag}h"
        frame[col] = y.shift(lag)
        delays[col] = lag
    frame["lag_load_168h_smooth3"] = (y.shift(167) + y.shift(168) + y.shift(169)) / 3.0
    delays["lag_load_168h_smooth3"] = 167
    for w in ROLL_WINDOWS_H:
        rolled = y.rolling(w)
        for stat in ROLL_STATS:
            col = f"roll_{stat}_{w}h"
            frame[col] = getattr(rolled, stat)().shift(horizon)
            delays[col] = horizon
    if prevision_j1 is not None:
        frame["lag_rte_prevision_j1"] = prevision_j1.reindex(index)
        delays["lag_rte_prevision_j1"] = RTE_D1_DELAY_H

    out = pd.DataFrame(frame, index=index)
    return FeatureBlock("lags", out, delays, reads_target=True)
