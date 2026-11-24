"""Mini end-to-end backtest on the real processed data (baselines only, 2 folds)."""

from __future__ import annotations

from hexaload.backtest.run import BacktestConfig, run
from hexaload.tuning.spaces import FEATURE_MODELS, TSFM_MODELS, UNIVARIATE_MODELS

ALL_MODELS = ",".join(FEATURE_MODELS + UNIVARIATE_MODELS + TSFM_MODELS)


def test_mini_backtest_baselines_only(tmp_path):
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
    # RTE reference beats the naive baselines on these folds
    mase = dict(zip(table["model"], table["mase_mean"], strict=False))
    assert mase["rte_prevision_j1"] < mase["naive"]
