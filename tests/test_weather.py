"""Tests for the Open-Meteo (weather) side of fetch + validate — offline."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
import requests

from hexaload.data import fetch
from hexaload.data.fetch import (
    OPENMETEO_HOURLY,
    fetch_openmeteo_dataset,
    fetch_openmeteo_month,
    france_grid,
    load_manifest,
    read_cache,
)
from hexaload.data.validate import (
    WeatherRecord,
    aggregate_weather_national,
    build_weather_dataset,
    validate_weather,
)


def _fake_response(payload) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode()
    response.headers["Content-Type"] = "application/json"
    return response


def _hourly_payload(lat: float, lon: float) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": {
            "time": ["2023-06-01T00:00", "2023-06-01T01:00"],
            "temperature_2m": [12.0, 11.0],
            "apparent_temperature": [10.0, 9.0],
            "relative_humidity_2m": [80.0, 82.0],
            "pressure_msl": [1013.0, 1012.0],
            "precipitation": [0.0, 0.2],
            "wind_speed_10m": [5.0, 6.0],
            "shortwave_radiation": [0.0, 0.0],
            "cloud_cover": [20.0, 30.0],
            "weather_code": [1, 3],
            "is_day": [0, 0],
        },
    }


def test_france_grid_size_and_bounds():
    grid = france_grid(1.0)
    assert len(grid) == 11 * 14  # lat 41.5..51.5, lon -5..8, step 1
    assert grid[0] == (41.5, -5.0)
    assert grid[-1] == (51.5, 8.0)


def test_openmeteo_hourly_covers_the_full_weather_set():
    for var in [
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "precipitation",
        "wind_speed_10m",
        "shortwave_radiation",
        "cloud_cover",
        "weather_code",
        "is_day",
        "apparent_temperature",
    ]:
        assert var in OPENMETEO_HOURLY.split(",")


def test_weather_record_pins_naive_times_to_utc():
    rec = WeatherRecord.model_validate(
        {
            "time": "2023-06-01T00:00",
            "latitude": 48.85,
            "longitude": 2.35,
            "temperature_2m": 12.5,
            "pressure_msl": 1013.2,
            "weather_code": 1,
            "is_day": 0,
            "model": "icon_eu",
        }
    )
    assert rec.time.tzinfo == UTC
    assert rec.pressure_msl == 1013.2
    assert rec.weather_code == 1


def test_validate_weather_coerces_nan_and_strings():
    raw = pd.DataFrame(
        [
            {
                "time": "2023-06-01T00:00",
                "latitude": 48.85,
                "longitude": 2.35,
                "temperature_2m": float("nan"),
                "relative_humidity_2m": "80.5",
                "pressure_msl": 1013.0,
                "weather_code": "95",
                "model": "icon_eu",
            },
            {
                "time": "2023-06-01T01:00",
                "latitude": 48.85,
                "longitude": 2.35,
                "temperature_2m": "13.5",
                "relative_humidity_2m": 81.0,
                "pressure_msl": None,
                "model": "icon_eu",
            },
        ]
    )
    clean, report = validate_weather(raw)
    assert report.valid_rows == 2
    assert pd.isna(clean.loc[0, "temperature_2m"])
    assert clean.loc[0, "relative_humidity_2m"] == 80.5
    assert clean.loc[0, "weather_code"] == 95
    assert clean.loc[1, "temperature_2m"] == 13.5
    assert pd.isna(clean.loc[1, "pressure_msl"])


def test_aggregate_weather_national_mean_and_mode():
    clean = pd.DataFrame(
        {
            "time": ["2023-06-01T00:00", "2023-06-01T00:00", "2023-06-01T01:00"],
            "temperature_2m": [10.0, 20.0, 15.0],
            "precipitation": [0.0, 1.0, 0.5],
            "weather_code": [1, 3, 3],
            "is_day": [0, 0, 1],
        }
    )
    hourly = aggregate_weather_national(clean)
    assert "temperature_2m_c" in hourly.columns
    assert "precipitation_mm" in hourly.columns
    assert "is_day_fraction" in hourly.columns
    assert "weather_code_code" in hourly.columns
    assert list(hourly["temperature_2m_c"]) == [15.0, 15.0]
    # mode of [1, 3] is ambiguous -> first mode (1); of [3] -> 3
    assert hourly["weather_code_code"].iloc[0] in (1, 3)
    assert hourly["weather_code_code"].iloc[1] == 3
    assert list(hourly["is_day_fraction"]) == [0.0, 1.0]
    assert hourly.index.name == "date_heure_utc"


def test_fetch_openmeteo_month_parses_multi_var_payload(monkeypatch):
    client = fetch.HttpClient(base_url=fetch.OPENMETEO_BASE, min_interval_s=0.0)
    state: dict[str, dict] = {"params": {}}

    def fake_get(url, params=None, timeout=None):
        state["params"] = params or {}
        return _fake_response(
            [_hourly_payload(48.875, 2.375), _hourly_payload(46.25, -1.5)]
        )

    monkeypatch.setattr(client.session, "get", fake_get)
    records = fetch_openmeteo_month(
        client,
        [(48.875, 2.375), (46.25, -1.5)],
        datetime(2023, 6, 1, tzinfo=UTC),
        datetime(2023, 7, 1, tzinfo=UTC),
    )
    assert len(records) == 4
    assert records[0]["model"] == "icon_eu"
    assert records[0]["temperature_2m"] == 12.0
    assert records[0]["relative_humidity_2m"] == 80.0
    assert records[0]["pressure_msl"] == 1013.0
    assert records[0]["precipitation"] == 0.0
    assert records[0]["weather_code"] == 1
    # the request asks for the full hourly variable set
    for var in OPENMETEO_HOURLY.split(","):
        assert var in state["params"]["hourly"]


def test_fetch_openmeteo_dataset_caches_month(tmp_path, monkeypatch):
    client = fetch.HttpClient(base_url=fetch.OPENMETEO_BASE, min_interval_s=0.0)

    def fake_get(url, params=None, timeout=None):
        params = params or {}
        lats = params["latitude"].split(",")
        lons = params["longitude"].split(",")
        payload = [
            _hourly_payload(float(a), float(b)) for a, b in zip(lats, lons, strict=True)
        ]
        return _fake_response(payload)

    monkeypatch.setattr(client.session, "get", fake_get)
    # tiny grid: shrink the module constants for the test
    monkeypatch.setattr(fetch, "GRID_LAT_RANGE", (48.0, 49.0))
    monkeypatch.setattr(fetch, "GRID_LON_RANGE", (2.0, 3.0))
    files = fetch_openmeteo_dataset(
        datetime(2023, 6, 1, tzinfo=UTC),
        datetime(2023, 7, 1, tzinfo=UTC),
        tmp_path,
        client=client,
    )
    assert len(files) == 1
    manifest = load_manifest(tmp_path / "openmeteo-icon_eu")
    assert manifest["model"] == "icon_eu"
    assert manifest["ranges"]["2023-06"]["points"] == 4  # 2x2 grid
    df = read_cache("openmeteo-icon_eu", tmp_path)
    assert len(df) == 8  # 4 grid points x 2 hourly timesteps
    assert "pressure_msl" in df.columns
    assert "precipitation" in df.columns


def test_build_weather_dataset_end_to_end():
    raw = pd.DataFrame(
        {
            "time": ["2023-06-01T00:00", "2023-06-01T00:00"],
            "latitude": [48.85, 46.25],
            "longitude": [2.35, -1.5],
            "temperature_2m": [10.0, 20.0],
            "relative_humidity_2m": [60.0, 80.0],
            "pressure_msl": [1000.0, 1020.0],
            "precipitation": [0.0, 2.0],
            "model": ["icon_eu", "icon_eu"],
        }
    )
    hourly, report = build_weather_dataset(raw)
    assert report.valid_rows == 2
    assert hourly.iloc[0]["temperature_2m_c"] == pytest.approx(15.0)
    assert hourly.iloc[0]["relative_humidity_2m_pct"] == pytest.approx(70.0)
    assert hourly.iloc[0]["pressure_msl_hpa"] == pytest.approx(1010.0)
    assert hourly.iloc[0]["precipitation_mm"] == pytest.approx(1.0)
