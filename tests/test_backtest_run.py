"""Mini end-to-end backtest on SYNTHETIC data (no data/processed needed -> CI-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hexaload.backtest import load as backtest_load
from hexaload.backtest.run import BacktestConfig, run
from hexaload.tuning.spaces import FEATURE_MODELS, TSFM_MODELS, UNIVARIATE_MODELS

ALL_MODELS = ",".join(FEATURE_MODELS + UNIVARIATE_MODELS + TSFM_MODELS)


@pytest.fixture
def synthetic_processed(tmp_path, monkeypatch):
    """Fake data/processed: hourly cons-def + weather, no TR tail, no features."""
    idx = pd.date_range("2026-10-01", periods=31 * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    t = np.arange(len(idx))
    conso = 50_000 + 5_000 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 50, len(idx))
    cons_def = pd.DataFrame(
        {
            "consommation": conso,
            # D-1 forecast = consumption 24h earlier (near-perfect on this signal)
            "prevision_j1": np.roll(conso, 24) + rng.normal(0, 30, len(idx)),
        },
        index=idx,
    )
    cons_def.index.name = "date_heure"
    cons_def.to_parquet(tmp_path / "eco2mix-national-cons-def-hourly.parquet")
    pd.DataFrame({"t2m": np.zeros(len(idx))}, index=idx).to_parquet(
        tmp_path / "openmeteo-icon_eu-hourly-national.parquet"
    )
    monkeypatch.setattr(
        backtest_load,
        "CONS_DEF_PARQUET",
        tmp_path / "eco2mix-national-cons-def-hourly.parquet",
    )
    monkeypatch.setattr(
        backtest_load,
        "WX_PARQUET",
        tmp_path / "openmeteo-icon_eu-hourly-national.parquet",
    )
    # absent on purpose: no real-time tail, no feature table -> baselines only
    monkeypatch.setattr(backtest_load, "TR_PARQUET", tmp_path / "missing-tr.parquet")
    monkeypatch.setattr(
        backtest_load, "FEATURES_PARQUET", tmp_path / "missing-features.parquet"
    )
    return tmp_path


def test_mini_backtest_baselines_only(tmp_path, synthetic_processed):
    cfg = BacktestConfig(
        start="2026-10-01",
        end="2026-11-01",
        horizon=24,
        window_length=24 * 20,
        step_length=24 * 5,
        skip=ALL_MODELS,
        out=str(tmp_path / "bt.parquet"),
    )
    table = run(cfg)
    assert set(table["model"]) == {
        "naive",
        "seasonal_naive_24",
        "seasonal_naive_168",
        "rte_prevision_j1",
    }
    assert (tmp_path / "bt.parquet").exists()
    assert (tmp_path / "bt.md").exists()
    # the near-perfect RTE reference beats the naive baseline on these folds
    mase = dict(zip(table["model"], table["mase_mean"], strict=True))
    assert mase["rte_prevision_j1"] < mase["naive"]


def test_backtest_long_horizon_d1_references_scored_on_24h(
    tmp_path, synthetic_processed
):
    """h=48 (D+2-style): seasonal-naive sp=24 would read the future beyond 24 h —
    it and the RTE reference must be scored on the first 24 h of each block,
    without tripping the anti-leakage assertion."""
    cfg = BacktestConfig(
        start="2026-10-01",
        end="2026-11-01",
        horizon=48,
        window_length=24 * 20,
        step_length=24 * 5,
        skip=ALL_MODELS,
        out=str(tmp_path / "bt48.parquet"),
    )
    table = run(cfg)
    assert set(table["model"]) == {
        "naive",
        "seasonal_naive_24",
        "seasonal_naive_168",
        "rte_prevision_j1",
    }
    assert table["mase_mean"].notna().all()
