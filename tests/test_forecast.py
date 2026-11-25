"""Forecast script on SYNTHETIC data (CI-safe: no data/processed, no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from hexaload.backtest import load as backtest_load

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


forecast = load_script("forecast")


@pytest.fixture
def synthetic_processed(tmp_path, monkeypatch):
    """Fake data/processed: cons-def + weather + a tiny feature table."""
    idx = pd.date_range("2026-10-01", periods=31 * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    t = np.arange(len(idx))
    conso = 50_000 + 5_000 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 50, len(idx))
    cons_def = pd.DataFrame(
        {"consommation": conso, "prevision_j1": np.roll(conso, 24)}, index=idx
    )
    cons_def.index.name = "date_heure"
    cons_def.to_parquet(tmp_path / "cons-def.parquet")
    wx = pd.DataFrame(
        {
            "temperature_2m_c": np.full(len(idx), 10.0),
            "relative_humidity_2m_pct": np.full(len(idx), 60.0),
        },
        index=idx,
    )
    wx.index.name = "date_heure_utc"
    wx.to_parquet(tmp_path / "wx.parquet")
    # build the feature table with the REAL blocks so column names match
    from hexaload.features.table import build_feature_table

    feats, manifest = build_feature_table(
        idx,
        wx=wx,
        y=pd.Series(conso, index=idx),
        prevision_j1=pd.Series(np.roll(conso, 24), index=idx),
        horizon=24,
    )
    feats.to_parquet(tmp_path / "features.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({"features": manifest}))
    monkeypatch.setattr(
        backtest_load, "CONS_DEF_PARQUET", tmp_path / "cons-def.parquet"
    )
    monkeypatch.setattr(backtest_load, "WX_PARQUET", tmp_path / "wx.parquet")
    monkeypatch.setattr(backtest_load, "TR_PARQUET", tmp_path / "missing-tr.parquet")
    monkeypatch.setattr(
        backtest_load, "FEATURES_PARQUET", tmp_path / "features.parquet"
    )
    monkeypatch.setattr(backtest_load, "FEATURES_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(forecast, "OUT_DIR", tmp_path / "forecasts")
    return tmp_path


def test_freeze_as_of_is_1200_paris_day_before():
    # summer (CEST = UTC+2): 12:00 Paris = 10:00 UTC
    target = cast(pd.Timestamp, pd.Timestamp("2026-07-10", tz="UTC"))
    from hexaload.backtest.recursive import freeze_as_of

    as_of = freeze_as_of(target)
    assert as_of == pd.Timestamp("2026-07-09 10:00", tz="UTC")
    # winter (CET = UTC+1): 12:00 Paris = 11:00 UTC
    target = cast(pd.Timestamp, pd.Timestamp("2026-01-10", tz="UTC"))
    assert freeze_as_of(target) == pd.Timestamp("2026-01-09 11:00", tz="UTC")


def test_default_target_depends_on_paris_noon():
    morning = pd.Timestamp("2026-07-09 08:00", tz="UTC")  # 10:00 Paris -> tomorrow
    assert forecast.default_target(morning).date() == pd.Timestamp("2026-07-10").date()
    evening = pd.Timestamp("2026-07-09 12:00", tz="UTC")  # 14:00 Paris -> +2 days
    assert forecast.default_target(evening).date() == pd.Timestamp("2026-07-11").date()


def test_forecast_writes_json_and_is_idempotent(
    tmp_path, synthetic_processed, monkeypatch
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "forecast.py",
            "--date",
            "2026-10-20",
            "--models",
            "ridge_features",
            "--params",
            str(tmp_path / "missing-params.yaml"),
            "--no-live-weather",
        ],
    )
    assert forecast.main() == 0
    out = tmp_path / "forecasts" / "2026-10-20.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["date"] == "2026-10-20"
    assert payload["as_of"] == "2026-10-19 10:00:00+00:00"  # CEST
    m = payload["models"]["ridge_features"]
    assert len(m["median"]) == 24 and len(m["median_d7"]) == 168
    assert len(m["q05"]) == 24 and len(m["q95"]) == 24
    # band is ordered
    assert all(a <= b for a, b in zip(m["q05"], m["q95"], strict=False))
    # idempotent: second run without --force does not rewrite
    mtime = out.stat().st_mtime_ns
    monkeypatch.setattr(
        "sys.argv",
        [
            "forecast.py",
            "--date",
            "2026-10-20",
            "--models",
            "ridge_features",
            "--no-live-weather",
        ],
    )
    assert forecast.main() == 0
    assert out.stat().st_mtime_ns == mtime


def test_forecast_fails_clearly_when_features_do_not_cover_horizon(
    tmp_path, synthetic_processed, monkeypatch
):
    # target far beyond the synthetic feature table (ends 2026-10-31)
    monkeypatch.setattr(
        "sys.argv",
        [
            "forecast.py",
            "--date",
            "2026-12-25",
            "--models",
            "ridge_features",
            "--no-live-weather",
        ],
    )
    with pytest.raises(SystemExit, match="NaN exogenous"):
        forecast.main()


def test_month_key_is_current():
    from datetime import UTC, datetime

    from hexaload.data.fetch import month_key_is_current

    now = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)
    assert month_key_is_current("2026-11", now)
    assert not month_key_is_current("2026-10", now)
    assert not month_key_is_current("2026-12", now)
