"""Lags block: delays >= horizon + future-perturbation no-leakage test."""

from typing import cast

import pandas as pd
import pytest

from hexaload.features import assert_horizon_safe
from hexaload.features.lags import lags_block


def test_direct_lags_and_delays(
    hourly_idx: pd.DatetimeIndex, y_load: pd.Series
) -> None:
    b = lags_block(y_load, index=hourly_idx, horizon=24)
    f = b.frame
    y = pd.Series(y_load.to_numpy(), index=hourly_idx)
    for lag in (24, 48, 168, 336):
        pd.testing.assert_series_equal(
            f[f"lag_load_{lag}h"], cast(pd.Series, y.shift(lag)), check_names=False
        )
        assert b.delays[f"lag_load_{lag}h"] == lag
    assert b.delays["lag_load_168h_smooth3"] == 167


def test_rolling_shifted_by_horizon(
    hourly_idx: pd.DatetimeIndex, y_load: pd.Series
) -> None:
    horizon = 24
    b = lags_block(y_load, index=hourly_idx, horizon=horizon)
    y = pd.Series(y_load.to_numpy(), index=hourly_idx)
    pd.testing.assert_series_equal(
        b.frame["roll_mean_24h"],
        cast(pd.Series, y.rolling(24).mean().shift(horizon)),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        b.frame["roll_std_720h"],
        cast(pd.Series, y.rolling(720).std().shift(horizon)),
        check_names=False,
    )
    assert b.delays["roll_mean_24h"] == horizon
    assert b.delays["roll_max_168h"] == horizon


def test_rte_prevision_j1_delay(
    hourly_idx: pd.DatetimeIndex, y_load: pd.Series
) -> None:
    pj1 = y_load * 0.98
    b = lags_block(y_load, index=hourly_idx, prevision_j1=pj1, horizon=24)
    assert b.delays["lag_rte_prevision_j1"] == 24
    pd.testing.assert_series_equal(
        b.frame["lag_rte_prevision_j1"],
        pd.Series(pj1.to_numpy(), index=hourly_idx),
        check_names=False,
    )


def test_manifest_horizon_safety(
    hourly_idx: pd.DatetimeIndex, y_load: pd.Series
) -> None:
    b = lags_block(y_load, index=hourly_idx, horizon=24)
    manifest = {c: {"block": "lags", "delay_h": d} for c, d in b.delays.items()}
    assert_horizon_safe(manifest, horizon=24)  # ok
    with pytest.raises(ValueError, match="leakage risk"):
        assert_horizon_safe(manifest, horizon=200)  # lag24 < 200h -> refuse


def test_future_perturbation_does_not_leak(
    hourly_idx: pd.DatetimeIndex, y_load: pd.Series
) -> None:
    """Overwriting y AFTER as_of must not change any available lag feature."""
    as_of = hourly_idx[24 * 30]
    y_future = y_load.copy()
    y_future.loc[as_of + pd.Timedelta(hours=1) :] = 1e9
    base = lags_block(y_load, index=hourly_idx, horizon=24).apply_as_of(as_of)
    pert = lags_block(y_future, index=hourly_idx, horizon=24).apply_as_of(as_of)
    pd.testing.assert_frame_equal(base, pert)
