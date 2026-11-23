"""Tests for the naive baselines and the RTE reference (offline)."""

import numpy as np
import pandas as pd
import pytest

from hexaload.backtest.baselines import (
    predict_naive,
    predict_rte_prevision_j1,
    predict_seasonal_naive,
)


def hourly(vals: list[float], start: str = "2023-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(vals), freq="1h", tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


def test_naive_carries_last_train_value() -> None:
    y = hourly([float(i) for i in range(100)])
    train = pd.DatetimeIndex(y.index[:80])
    test = pd.DatetimeIndex(y.index[80:104])
    pred = predict_naive(y.loc[train], test)
    assert (pred == 79.0).all()
    assert list(pred.index) == list(test)


def test_seasonal_naive_uses_lag_sp() -> None:
    y = hourly([float(i % 24) for i in range(24 * 10)])
    train = pd.DatetimeIndex(y.index[: 24 * 8])
    test = pd.DatetimeIndex(y.index[24 * 8 : 24 * 8 + 24])
    pred = predict_seasonal_naive(y, train, test, sp=24)
    expected = pd.Series(y.to_numpy()[24 * 8 - 24 : 24 * 8], index=test, name="y_pred")
    pd.testing.assert_series_equal(pred, expected)


def test_seasonal_naive_raises_without_history() -> None:
    y = hourly([1.0] * 30)
    train = pd.DatetimeIndex(y.index[:24])
    test = pd.DatetimeIndex(y.index[10:16])  # starts before sp -> no source
    with pytest.raises(ValueError, match="not enough history"):
        predict_seasonal_naive(y, train, test, sp=24)


def test_rte_reference_reindexes_on_test() -> None:
    ref = hourly([float(1000 + i) for i in range(100)])
    test = pd.DatetimeIndex(ref.index[90:96])
    pred = predict_rte_prevision_j1(ref, test)
    assert list(pred.to_numpy()) == [1090.0, 1091.0, 1092.0, 1093.0, 1094.0, 1095.0]
    # missing timestamps -> NaN (dropped later by the metrics alignment)
    pred_gap = predict_rte_prevision_j1(
        ref, pd.DatetimeIndex([test[0], test[0] + pd.Timedelta(days=365)], tz="UTC")
    )
    assert np.isnan(pred_gap.iloc[1])
