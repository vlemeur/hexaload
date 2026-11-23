"""Tests for the Open-Meteo (weather) side of fetch + validate — offline."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
import requests

from hexaload.data import fetch
from hexaload.data.fetch import (
    fetch_openmeteo_dataset,
    fetch_openmeteo_month,
    france_grid,
    load_manifest,
    read_cache,
)
from hexaload.data.validate import (
    WeatherRecord,
    aggregate_temperature_national,
    build_weather_dataset,
    validate_weather,
)


def _fake_response(payload) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode()
    response.headers["Content-Type"] = "application/json"
    return response


def test_france_grid_size_and_bounds():
    grid = france_grid(1.0)
    assert len(grid) == 11 * 14  # lat 41.5..51.5, lon -5..8, step 1
    assert grid[0] == (41.5, -5.0)
    assert grid[-1] == (51.5, 8.0)


def test_weather_record_pins_naive_times_to_utc():
    rec = WeatherRecord.model_validate(
        {
            "time": "2023-06-01T00:00",
            "latitude": 48.85,
            "longitude": 2.35,
            "temperature_2m": 12.5,
            "model": "icon_eu",
        }
    )
    assert rec.time.tzinfo == UTC


def test_validate_weather_coerces_nan_and_strings():
    raw = pd.DataFrame(
        [
            {
                "time": "2023-06-01T00:00",
                "latitude": 48.85,
                "longitude": 2.35,
                "temperature_2m": float("nan"),
                "model": "icon_eu",
            },
            {
                "time": "2023-06-01T01:00",
                "latitude": 48.85,
                "longitude": 2.35,
                "temperature_2m": "13.5",
                "model": "icon_eu",
            },
        ]
    )
    clean, report = validate_weather(raw)
    assert report.valid_rows == 2
    assert clean.loc[0, "temperature_2m"] is None or pd.isna(
        clean.loc[0, "temperature_2m"]
    )
    assert clean.loc[1, "temperature_2m"] == 13.5


def test_aggregate_temperature_national_means_grid_per_hour():
    clean = pd.DataFrame(
        {
            "time": ["2023-06-01T00:00", "2023-06-01T00:00", "2023-06-01T01:00"],
            "temperature_2m": [10.0, 20.0, 15.0],
        }
    )
    hourly = aggregate_temperature_national(clean)
    assert list(hourly["temperature_2m_c"]) == [15.0, 15.0]
    assert hourly.index.name == "date_heure_utc"


def test_fetch_openmeteo_month_parses_multi_point_payload(monkeypatch):
    client = fetch.HttpClient(base_url=fetch.OPENMETEO_BASE, min_interval_s=0.0)
    payload = [
        {
            "latitude": 48.875,
            "longitude": 2.375,
            "hourly": {
                "time": ["2023-06-01T00:00", "2023-06-01T01:00"],
                "temperature_2m": [12.0, 11.0],
            },
        },
        {
            "latitude": 46.25,
            "longitude": -1.5,
            "hourly": {
                "time": ["2023-06-01T00:00", "2023-06-01T01:00"],
                "temperature_2m": [14.0, 13.0],
            },
        },
    ]
    monkeypatch.setattr(
        client.session,
        "get",
        lambda url, params=None, timeout=None: _fake_response(payload),
    )
    records = fetch_openmeteo_month(
        client,
        [(48.875, 2.375), (46.25, -1.5)],
        datetime(2023, 6, 1, tzinfo=UTC),
        datetime(2023, 7, 1, tzinfo=UTC),
    )
    assert len(records) == 4
    assert records[0]["model"] == "icon_eu"
    assert records[0]["time"] == "2023-06-01T00:00"


def test_fetch_openmeteo_dataset_caches_month(tmp_path, monkeypatch):
    client = fetch.HttpClient(base_url=fetch.OPENMETEO_BASE, min_interval_s=0.0)

    def fake_get(url, params=None, timeout=None):
        params = params or {}
        lats = params["latitude"].split(",")
        lons = params["longitude"].split(",")
        payload = [
            {
                "latitude": float(a),
                "longitude": float(b),
                "hourly": {"time": ["2023-06-01T00:00"], "temperature_2m": [12.0]},
            }
            for a, b in zip(lats, lons, strict=True)
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
    assert len(df) == 4


def test_build_weather_dataset_end_to_end():
    raw = pd.DataFrame(
        {
            "time": ["2023-06-01T00:00", "2023-06-01T00:00"],
            "latitude": [48.85, 46.25],
            "longitude": [2.35, -1.5],
            "temperature_2m": [10.0, 20.0],
            "model": ["icon_eu", "icon_eu"],
        }
    )
    hourly, report = build_weather_dataset(raw)
    assert report.valid_rows == 2
    assert hourly.iloc[0]["temperature_2m_c"] == pytest.approx(15.0)
