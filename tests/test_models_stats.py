"""sktime statistical forecasters (ETS/Theta) — offline tests."""

import warnings

import numpy as np
import pandas as pd
import pytest

from hexaload.models import SktimeAutoETS, SktimeTheta


def make_y(n: int = 24 * 21) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    y = 50_000 + 3_000 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 100, n)
    return pd.Series(y, index=idx, name="consommation")


@pytest.mark.parametrize("cls", [SktimeAutoETS, SktimeTheta])
def test_predict_covers_test_index(cls: type) -> None:
    y = make_y()
    train, test = y.iloc[:-24], pd.DatetimeIndex(y.index[-24:])
    pred = cls().fit(train).predict(test)
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(test)
    assert pred.notna().all()
    assert (pred > 0).all()


def test_contract() -> None:
    y = make_y()
    with pytest.raises(RuntimeError, match="fit"):
        SktimeTheta().predict(pd.DatetimeIndex(y.index[-24:]))


def test_hourly_grid_fills_holes_and_sets_freq() -> None:
    from hexaload.models.stats import _hourly_grid

    idx = pd.date_range("2025-01-01", periods=48, freq="1h", tz="UTC")
    y = pd.Series(np.arange(48, dtype=float), index=idx)  # linear series
    y = y.drop(y.index[10])  # hole -> freq becomes None
    assert pd.DatetimeIndex(y.index).freq is None
    out = _hourly_grid(y)
    assert len(out) == 48
    assert pd.DatetimeIndex(out.index).freqstr == "h"
    assert out.iloc[10] == 10.0  # linear series: hole interpolated exactly


def test_fit_handles_hole_and_freq_none() -> None:
    y = make_y()
    y = y.drop(y.index[100])  # hole -> freq None, like the stitched eco2mix index
    train, test = y.iloc[:-24], pd.DatetimeIndex(y.index[-24:])
    pred = SktimeTheta().fit(train).predict(test)
    assert pred.index.equals(test)
    assert pred.notna().all()


def test_no_timezone_drop_warning_and_tz_preserved() -> None:
    y = make_y()
    train, test = y.iloc[:-24], pd.DatetimeIndex(y.index[-24:])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pred = SktimeTheta().fit(train).predict(test)
    assert not [w for w in rec if "timezone" in str(w.message).lower()]
    assert pred.index.equals(test)  # tz-aware UTC index preserved on output
    assert str(pd.DatetimeIndex(pred.index).tz) == "UTC"
