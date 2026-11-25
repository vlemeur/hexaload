"""Weather block: derivatives, thresholds, delays, weighted aggregation."""

import pandas as pd
import pytest

from hexaload.features.weather import (
    CDD_BASE_C,
    HDD_BASE_C,
    national_weather_frame,
    weather_block,
)


def test_hdd_cdd_thresholds(
    hourly_idx: pd.DatetimeIndex, wx_national: pd.DataFrame
) -> None:
    b = weather_block(wx_national, index=hourly_idx)
    f = b.frame
    t = f["wx_t_c"]
    pd.testing.assert_series_equal(
        f["wx_hdd18"], (HDD_BASE_C - t).clip(lower=0.0), check_names=False
    )
    pd.testing.assert_series_equal(
        f["wx_cdd24"], (t - CDD_BASE_C).clip(lower=0.0), check_names=False
    )
    # archived forecasts have real delays (run issued at t-k -> delay k)
    assert b.delays["wx_t_c"] == 0
    assert b.delays["wx_dp_3h"] == 3
    assert b.delays["wx_t_lag12h"] == 12
    assert b.delays["wx_t_lag24h"] == 24
    assert b.delays["wx_t_min_day"] == 24


def test_dew_point_magnus() -> None:
    idx = pd.date_range("2025-06-01", periods=3, freq="1h", tz="UTC")
    wx = pd.DataFrame(
        {
            "temperature_2m_c": [20.0, 20.0, 30.0],
            "apparent_temperature_c": [19.0, 19.0, 29.0],
            "relative_humidity_2m_pct": [50.0, 100.0, 50.0],
            "pressure_msl_hpa": [1013.0] * 3,
        },
        index=idx,
    )
    f = weather_block(wx, index=idx).frame
    # Magnus: T=20, RH=50% -> Td ≈ 9.3 °C ; RH=100% -> Td = T
    assert f["wx_dewpoint_c"].iloc[0] == pytest.approx(9.27, abs=0.05)
    assert f["wx_dewpoint_c"].iloc[1] == pytest.approx(20.0, abs=0.05)
    assert f["wx_dewpoint_c"].iloc[2] == pytest.approx(18.4, abs=0.1)


def test_day_min_max_amp_and_pressure_gradient(
    hourly_idx: pd.DatetimeIndex, wx_national: pd.DataFrame
) -> None:
    f = weather_block(wx_national, index=hourly_idx).frame
    t = f["wx_t_c"]
    local_day = hourly_idx.tz_convert("Europe/Paris").normalize()
    pd.testing.assert_series_equal(
        f["wx_t_min_day"],
        t.groupby(local_day).transform("min").shift(24),  # previous day: no leak
        check_names=False,
    )
    amp = f["wx_t_max_day"] - f["wx_t_min_day"]
    pd.testing.assert_series_equal(
        f["wx_t_amp_day"].dropna(), amp.dropna(), check_names=False
    )
    pd.testing.assert_series_equal(
        f["wx_dp_3h"], f["wx_p_hpa"].diff(3), check_names=False
    )


def test_weather_lags(hourly_idx: pd.DatetimeIndex, wx_national: pd.DataFrame) -> None:
    f = weather_block(wx_national, index=hourly_idx).frame
    for lag in (12, 24):
        pd.testing.assert_series_equal(
            f[f"wx_t_lag{lag}h"], f["wx_t_c"].shift(lag), check_names=False
        )
        pd.testing.assert_series_equal(
            f[f"wx_hdd18_lag{lag}h"], f["wx_hdd18"].shift(lag), check_names=False
        )


def test_missing_required_column_raises(
    hourly_idx: pd.DatetimeIndex, wx_national: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        weather_block(wx_national.drop(columns=["temperature_2m_c"]), index=hourly_idx)


def test_weighted_aggregation_uses_weights() -> None:
    # two grid points, all weight on point A -> national frame = point A values
    times = pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC")
    rows = []
    for i, t in enumerate(times):
        rows.append(
            {
                "time": t,
                "latitude": 45.0,
                "longitude": 2.0,
                "model": "m",
                "temperature_2m": 10.0 + i,
                "relative_humidity_2m": 50.0,
                "weather_code": 1,
            }
        )
        rows.append(
            {
                "time": t,
                "latitude": 48.0,
                "longitude": 5.0,
                "model": "m",
                "temperature_2m": 99.0,
                "relative_humidity_2m": 90.0,
                "weather_code": 3,
            }
        )
    raw = pd.DataFrame(rows)
    weights = pd.DataFrame(
        {"latitude": [45.0, 48.0], "longitude": [2.0, 5.0], "weight": [1.0, 0.0]}
    )
    out = national_weather_frame(raw, weights)
    assert list(out["temperature_2m_c"]) == [10.0, 11.0, 12.0, 13.0]
    assert list(out["weather_code_code"]) == [1, 1, 1, 1]
    # uniform fallback = plain mean
    out_mean = national_weather_frame(raw, None)
    assert list(out_mean["temperature_2m_c"]) == [
        (10.0 + i + 99.0) / 2 for i in range(4)
    ]
