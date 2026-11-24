"""Offline tests for the tuning package (spaces, build_model, tracking, hooks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hexaload.models import MLPFeatures
from hexaload.tuning.config import TuningConfig
from hexaload.tuning.spaces import TUNABLE, build_model, default_params, suggest_params
from hexaload.tuning.tracking import RunLogger, init_mlflow

optuna = pytest.importorskip("optuna")
mlflow = pytest.importorskip("mlflow")
yaml = pytest.importorskip("yaml")

ALL_MODELS = [
    "ridge_features",
    "hgb_features",
    "xgb_features",
    "mlp_features",
    "rnn_features",
    "tcn_features",
    "nhits",
    "patchtst_features",
    "sktime_autoets",
    "sktime_theta",
    "timestral_209m",
    "chronos_bolt",
]


def test_tunable_covers_all_feature_and_tsfm_models():
    for m in (
        "ridge_features",
        "hgb_features",
        "xgb_features",
        "mlp_features",
        "rnn_features",
        "tcn_features",
        "nhits",
        "patchtst_features",
        "timestral_209m",
        "chronos_bolt",
    ):
        assert m in TUNABLE


def test_build_model_every_name():
    for m in ALL_MODELS:
        model = build_model(m, default_params(m))
        assert model is not None
    with pytest.raises(ValueError):
        build_model("does_not_exist")


def test_suggest_params_within_bounds():
    sampler = optuna.samplers.TPESampler(seed=0)
    for m in TUNABLE:
        study = optuna.create_study(
            directions=["minimize", "minimize"], sampler=sampler
        )
        for _ in range(3):
            trial = study.ask()
            params = suggest_params(m, trial)
            study.tell(trial, (0.5, 500.0))  # dummy values, just to advance
            assert isinstance(params, dict)
            # every suggested value must be accepted by the constructor
            build_model(m, params)


def test_suggest_unknown_model_is_empty():
    study = optuna.create_study()
    trial = study.ask()
    assert suggest_params("naive", trial) == {}
    study.tell(trial, 1.0)


def test_run_logger_writes_local_file_store(tmp_path):
    cfg = TuningConfig(mlruns=str(tmp_path / "mlruns"), experiment="test-exp")
    uri = init_mlflow(cfg)
    assert uri.startswith("sqlite:///")
    with RunLogger(cfg, "test-exp", "run-1", {"model": "hgb_features"}) as rl:
        rl.params({"lr": 0.05, "hidden": (128, 64)})
        rl.metrics({"mase_mean": 0.4, "crps_mean": 800.0})
    exp = mlflow.get_experiment_by_name("test-exp")
    assert exp is not None
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["tags.model"] == "hgb_features"
    assert row["metrics.mase_mean"] == pytest.approx(0.4)
    assert row["metrics.wall_time_s"] >= 0.0
    assert row["metrics.peak_mb"] > 0.0


def _tiny_series(n=24 * 40):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(np.sin(np.arange(n) / 24 * 2 * np.pi) * 100 + 5000, index=idx)


def test_torch_on_epoch_end_hook_called():
    calls: list[tuple[int, float]] = []
    y = _tiny_series()
    model = MLPFeatures(
        h=24,
        hidden=(32,),
        max_epochs=3,
        patience=3,
        input_size=48,
        device="cpu",
        seed=0,
        on_epoch_end=lambda e, v: calls.append((e, v)),
    )
    model.fit(y)
    assert len(calls) >= 1  # called once per epoch actually run
    assert all(isinstance(e, int) and np.isfinite(v) for e, v in calls)
    epochs = [e for e, _ in calls]
    assert epochs == list(range(len(epochs)))  # 0,1,2,...


def test_torch_hook_can_abort_training():
    class _Stop(Exception):
        pass

    def hook(e, v):
        if e >= 1:
            raise _Stop

    y = _tiny_series()
    model = MLPFeatures(
        h=24,
        hidden=(32,),
        max_epochs=10,
        patience=10,
        input_size=48,
        device="cpu",
        seed=0,
        on_epoch_end=hook,
    )
    with pytest.raises(_Stop):
        model.fit(y)


# --- integration: mini Optuna study + hold-out on synthetic data -------------

from hexaload.backtest.walkforward import walk_forward_splits  # noqa: E402
from hexaload.tuning.evaluate import run_holdout  # noqa: E402
from hexaload.tuning.params import load_best_params, save_best_params  # noqa: E402
from hexaload.tuning.tune import (  # noqa: E402
    TuningData,
    pick_best_params,
    run_study,
    score_folds,
    select_folds,
)


def _synth_data(tmp_path) -> TuningData:
    n = 24 * 120
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    y = pd.Series(
        5000 + 100 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 20, n),
        index=idx,
    )
    splits = walk_forward_splits(
        y,
        start=idx[0],
        end=idx[-1],
        horizon=24,
        window_length=24 * 60,
        step_length=24 * 10,
    )
    return TuningData(
        y=y,
        X=None,
        manifest=None,
        exo_cols=[],
        splits=splits,
        tune_splits=splits[:2],
        holdout_splits=splits[2:4],
    )


def _cfg(tmp_path) -> TuningConfig:
    return TuningConfig(
        storage=str(tmp_path / "optuna.db"),
        mlruns=str(tmp_path / "mlruns"),
        out_params=str(tmp_path / "best_params.yaml"),
        budget_minutes=5.0,
        n_trials=2,
        seed=0,
    )


def test_select_folds_reserves_holdout():
    n = 24 * 160
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    y = pd.Series(np.arange(n, dtype=float), index=idx)
    splits = walk_forward_splits(
        y,
        start=idx[0],
        end=idx[-1],
        horizon=24,
        window_length=24 * 60,
        step_length=24 * 10,
    )
    cfg = TuningConfig(tune_folds=3, holdout_start="2024-04-15")
    tune, holdout = select_folds(splits, cfg)
    cut = pd.Timestamp("2024-04-15", tz="UTC")
    assert len(tune) == 3  # spread over the pre-holdout folds
    assert all(s.test_start < cut for s in tune)
    assert all(s.test_start >= cut for s in holdout)
    pre_starts = {s.test_start for s in splits if s.test_start < cut}
    split_starts = {s.test_start for s in splits}
    assert {t.test_start for t in tune} <= pre_starts
    assert {h.test_start for h in holdout} == split_starts - pre_starts


def test_mini_study_and_holdout(tmp_path):
    cfg = _cfg(tmp_path)
    data = _synth_data(tmp_path)
    study, best = run_study("sktime_theta", cfg, data)
    assert len(study.trials) == 2  # n_trials=2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert best == {}  # theta has no tunable params
    assert pick_best_params(study) == {}
    # study persisted in SQLite and resumable
    study2 = optuna.load_study(
        study_name=study.study_name, storage=f"sqlite:///{tmp_path / 'optuna.db'}"
    )
    assert len(study2.trials) == 2
    # best params YAML round-trip
    save_best_params({"sktime_theta": best}, cfg.out_params)
    assert load_best_params(cfg.out_params) == {"sktime_theta": {}}
    # hold-out scored once, records well-formed
    records = run_holdout("sktime_theta", best, cfg, data)
    assert len(records) == len(data.holdout_splits)
    assert all(np.isfinite(r["mase"]) and r["crps"] is not None for r in records)


def test_score_folds_univariate_directly(tmp_path):
    cfg = _cfg(tmp_path)
    data = _synth_data(tmp_path)
    records = score_folds("sktime_theta", {}, data.tune_splits, data, cfg)
    assert len(records) == 2
    assert set(records[0]) >= {"fold", "model", "mase", "crps"}
