"""Smoke test: build the materialized feature table from a fake cache (offline)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hexaload.features.build import build


def _make_cache(cache: Path) -> None:
    # eco2mix cons-def raw records (30-min grid, 10 days)
    times = pd.date_range("2025-01-01", periods=24 * 10 * 2, freq="30min", tz="UTC")
    eco = pd.DataFrame(
        {
            "date_heure": times,
            "date": times.tz_localize(None).date.astype(str),
            "heure": times.tz_localize(None).strftime("%H:%M"),
            "perimetre": "France",
            "nature": "Données consolidées",
            "consommation": 50_000 + 100 * np.arange(len(times)),
            "prevision_j1": 49_000 + 100 * np.arange(len(times)),
            "prevision_j": 48_500 + 100 * np.arange(len(times)),
        }
    )
    d = cache / "eco2mix-national-cons-def"
    d.mkdir(parents=True)
    eco.to_parquet(d / "2025-01.parquet", index=False)
    (d / "manifest.json").write_text("{}")

    # openmeteo raw grid records (2 points x 10 days hourly)
    wrows = []
    for t in pd.date_range("2025-01-01", periods=24 * 10, freq="1h", tz="UTC"):
        for lat, lon in [(45.0, 2.0), (48.0, 5.0)]:
            wrows.append(
                {
                    "time": t,
                    "latitude": lat,
                    "longitude": lon,
                    "model": "icon_eu",
                    "temperature_2m": 10.0 + (t.hour % 24) * 0.5,
                    "relative_humidity_2m": 60.0,
                    "pressure_msl": 1013.0,
                    "weather_code": 1,
                }
            )
    w = cache / "openmeteo-icon_eu"
    w.mkdir(parents=True)
    pd.DataFrame(wrows).to_parquet(w / "2025-01.parquet", index=False)
    (w / "manifest.json").write_text("{}")


def test_build_writes_parquet_and_manifest(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    _make_cache(cache)
    parquet, manifest_path = build(cache, out, horizon=24)
    assert parquet.exists() and manifest_path.exists()
    assert (out / "features-latest.parquet").exists()
    assert (out / "features-latest.manifest.json").exists()

    X = pd.read_parquet(parquet)
    meta = json.loads(manifest_path.read_text())
    assert meta["rows"] == len(X) == 24 * 10
    assert meta["horizon_h"] == 24
    assert set(meta["blocks"]) == {"calendar", "cyclic", "weather", "lags"}
    assert X.index.equals(
        pd.DatetimeIndex(pd.date_range("2025-01-01", periods=240, freq="1h", tz="UTC"))
    )
    # every column has a delay in the manifest; lags >= horizon
    assert set(meta["features"]) == set(X.columns)
    for _col, m in meta["features"].items():
        if m["block"] == "lags":
            assert m["delay_h"] >= 24
    # spot-check a lag and a weather derivative
    assert "lag_load_168h" in X.columns
    assert "wx_hdd18" in X.columns
    # inputs hashed (tr absent -> None)
    assert meta["inputs_sha256"]["tr_manifest"] is None
    assert meta["inputs_sha256"]["school_vacations"]
