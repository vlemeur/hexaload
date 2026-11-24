"""Optuna tuning driver: multi-objective (MASE, CRPS) walk-forward mini-backtests.

Design (à la onomatron):
- persistent SQLite studies (``load_if_exists``) — a tuning run can be resumed;
- NSGA-II seeded sampler over the two objectives (minimize MASE, minimize CRPS);
- MedianPruner fed by per-epoch val-loss reports for torch models (ASHA-style);
- every completed trial is logged to MLflow (params + metrics + wall time);
- the "best" params are picked on the Pareto front: lowest MASE among trials
  whose CRPS is within 5% of the best CRPS seen;
- a wall-clock budget stops the study between trials.

Tuning folds are spread over the pre-holdout period; folds >= holdout_start
are reserved for :mod:`hexaload.tuning.evaluate` (touched once).
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from hexaload.backtest.features import fold_features
from hexaload.backtest.load import exogenous_columns, load_features, load_targets
from hexaload.backtest.walkforward import Split, walk_forward_splits
from hexaload.metrics import DEFAULT_SP, crps, mase, sample_quantiles
from hexaload.models import DEFAULT_QUANTILES
from hexaload.progress import BaseProgress, NullProgress
from hexaload.tuning.config import TuningConfig
from hexaload.tuning.spaces import (
    FEATURE_MODELS,
    TORCH_MODELS,
    TSFM_MODELS,
    build_model,
    suggest_params,
)
from hexaload.tuning.tracking import RunLogger, init_mlflow

logger = logging.getLogger(__name__)

N_SAMPLES = 100  # same ensemble size as run_backtest.py for the CRPS
PENALTY = 1e9  # returned when a trial produces no CRPS at all


@dataclass
class TuningData:
    """Everything the objective needs, loaded once per tuning session."""

    y: pd.Series
    X: pd.DataFrame | None
    manifest: dict | None
    exo_cols: list[str]
    splits: list[Split]
    tune_splits: list[Split]
    holdout_splits: list[Split]


def select_folds(
    splits: list[Split], cfg: TuningConfig
) -> tuple[list[Split], list[Split]]:
    """(tuning folds, hold-out folds). Tuning folds are spread evenly over the
    pre-holdout period; hold-out = folds whose test block starts >= holdout_start."""
    cut = pd.Timestamp(cfg.holdout_start, tz="UTC")
    pre = [s for s in splits if s.test_start < cut]
    holdout = [s for s in splits if s.test_start >= cut]
    if not pre:
        raise ValueError(f"no fold before holdout_start={cfg.holdout_start}")
    if cfg.tune_folds >= len(pre):
        tune = pre
    else:
        idx = np.unique(
            np.linspace(0, len(pre) - 1, cfg.tune_folds).round().astype(int)
        )
        tune = [pre[i] for i in idx]
    logger.info(
        "folds: %d tuning %s -> %s | %d hold-out %s -> %s",
        len(tune),
        tune[0].test_start.date(),
        tune[-1].test_start.date(),
        len(holdout),
        holdout[0].test_start.date() if holdout else "-",
        holdout[-1].test_start.date() if holdout else "-",
    )
    return tune, holdout


def load_tuning_data(cfg: TuningConfig) -> TuningData:
    y, _ref, _wx, _source = load_targets()
    X, manifest = load_features()
    splits = walk_forward_splits(
        y,
        start=cfg.start,
        end=cfg.end,
        horizon=cfg.horizon,
        window_length=cfg.window_length,
        step_length=cfg.step_length,
    )
    logger.info(
        "%d walk-forward folds (window=%dh, step=%dh)",
        len(splits),
        cfg.window_length,
        cfg.step_length,
    )
    tune, holdout = select_folds(splits, cfg)
    return TuningData(
        y=y,
        X=X,
        manifest=manifest,
        exo_cols=exogenous_columns(manifest),
        splits=splits,
        tune_splits=tune,
        holdout_splits=holdout,
    )


def score_folds(
    model: str,
    params: dict[str, Any],
    splits: list[Split],
    data: TuningData,
    cfg: TuningConfig,
    progress: BaseProgress | None = None,
) -> list[dict]:
    """One record (mase/crps) per fold for a model instantiated with ``params``."""
    progress = progress or NullProgress()
    records: list[dict] = []
    for split in splits:
        progress.context(fold=str(split.test_start.date()))
        y_train = data.y.loc[split.train]
        y_test = data.y.loc[split.test]
        X_fold = (
            fold_features(data.X, data.manifest, split)
            if data.X is not None and data.manifest is not None
            else None
        )
        m = build_model(model, params)
        q = None
        if model in FEATURE_MODELS:
            m.fit(y_train, X_fold.loc[split.train])
            pred = m.predict(split.test, X_fold.loc[split.test])
            with suppress(NotImplementedError):
                q = m.predict_quantiles(
                    split.test, X_fold.loc[split.test], DEFAULT_QUANTILES
                )
        elif model in TSFM_MODELS:
            X_tr = (
                X_fold.loc[split.train][data.exo_cols] if X_fold is not None else None
            )
            X_te = X_fold.loc[split.test][data.exo_cols] if X_fold is not None else None
            m.fit(y_train, X_tr)
            pred = m.predict(split.test, X_te)
            with suppress(NotImplementedError):
                q = m.predict_quantiles(split.test, X_te, DEFAULT_QUANTILES)
        else:  # univariate
            m.fit(y_train)
            pred = m.predict(split.test)
            with suppress(NotImplementedError):
                q = m.predict_quantiles(split.test, quantiles=DEFAULT_QUANTILES)
        rec = {
            "fold": str(split.test_start.date()),
            "model": model,
            "mase": mase(y_test, pred, y_train=y_train, sp=DEFAULT_SP),
        }
        if q is not None:
            samples = sample_quantiles(q, DEFAULT_QUANTILES, N_SAMPLES, seed=cfg.seed)
            rec["crps"] = crps(y_test, samples)
        else:
            rec["crps"] = None  # point-only model
        records.append(rec)
    return records


def _objective(model: str, cfg: TuningConfig, data: TuningData, progress: BaseProgress):
    """Optuna objective: mean MASE and mean CRPS over the tuning folds."""

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        progress.context(model=model, trial=f"{trial.number}")
        params = suggest_params(model, trial)
        if model in TORCH_MODELS:
            params.update(
                max_epochs=cfg.max_epochs_torch,
                device=cfg.device,
                seed=cfg.seed + trial.number,
            )

            def on_epoch_end(epoch: int, val_loss: float) -> None:
                trial.report(val_loss, step=epoch)
                progress.context(
                    epoch=f"{epoch}/{cfg.max_epochs_torch}", val=f"{val_loss:.4f}"
                )
                if trial.should_prune():
                    raise optuna.TrialPruned()

            params["on_epoch_end"] = on_epoch_end
        records = score_folds(model, params, data.tune_splits, data, cfg, progress)
        mase_mean = float(np.mean([r["mase"] for r in records]))
        crps_vals = [r["crps"] for r in records if r["crps"] is not None]
        crps_mean = float(np.mean(crps_vals)) if crps_vals else PENALTY
        return mase_mean, crps_mean

    return objective


def pick_best_params(study: optuna.Study) -> dict[str, Any]:
    """Lowest MASE on the Pareto front among trials within 5% of the best CRPS."""
    front = study.best_trials  # Pareto front for multi-objective studies
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not front:
        if not complete:
            raise RuntimeError("study has no completed trial")
        front = [min(complete, key=lambda t: t.values[0])]  # type: ignore[union-attr]
    min_crps = min(t.values[1] for t in front)  # type: ignore[index]
    cands = [t for t in front if t.values[1] <= min_crps * 1.05]  # type: ignore[index]
    best = min(cands, key=lambda t: t.values[0])  # type: ignore[index]
    return dict(best.params)


def run_study(
    model: str,
    cfg: TuningConfig,
    data: TuningData,
    progress: BaseProgress | None = None,
) -> tuple[optuna.Study, dict[str, Any]]:
    """Run (or resume) the Optuna study for one model; log trials to MLflow."""
    progress = progress or NullProgress()
    init_mlflow(cfg)
    Path(cfg.storage).parent.mkdir(parents=True, exist_ok=True)
    study_name = f"hexaload-{model}"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{Path(cfg.storage).resolve()}",
        directions=["minimize", "minimize"],  # (MASE, CRPS)
        sampler=optuna.samplers.NSGAIISampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=4),
        load_if_exists=True,
    )
    n_before = len(study.trials)
    t0 = time.perf_counter()
    # "+k/n": trials launched by THIS call (the study itself may be resumed)
    progress.context(model=model, trials=f"+{0}/{cfg.n_trials}")

    def on_trial_done(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # advance the global bar after every trial (complete or pruned)
        progress.context(
            model=model, trials=f"+{len(study.trials) - n_before}/{cfg.n_trials}"
        )
        if trial.state == optuna.trial.TrialState.COMPLETE:
            progress.advance(
                1,
                model=model,
                trial=f"{trial.number}",
                mase=f"{trial.values[0]:.4f}",  # type: ignore[index]
                crps=f"{trial.values[1]:.0f}",  # type: ignore[index]
            )
        else:
            progress.advance(
                1, model=model, trial=f"{trial.number}", state=trial.state.name.lower()
            )
        if time.perf_counter() - t0 > cfg.budget_minutes * 60:
            logger.info(
                "budget %.0f min exhausted — stopping study %s",
                cfg.budget_minutes,
                study_name,
            )
            study.stop()

    study.optimize(
        _objective(model, cfg, data, progress),
        n_trials=cfg.n_trials,
        callbacks=[on_trial_done],
        gc_after_trial=True,
        show_progress_bar=False,
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    logger.info(
        "%s: %d trials (%d new), %d complete, Pareto front %d",
        model,
        len(study.trials),
        len(study.trials) - n_before,
        len(done),
        len(study.best_trials),
    )
    for t in done:
        with RunLogger(
            cfg,
            cfg.experiment,
            f"{model}-trial-{t.number}",
            {"model": model, "trial": str(t.number)},
        ) as rl:
            rl.params({f"p.{k}": v for k, v in t.params.items()})
            rl.metrics({"mase_mean": t.values[0], "crps_mean": t.values[1]})  # type: ignore[index]
    best = pick_best_params(study)
    with RunLogger(
        cfg, cfg.experiment, f"{model}-best", {"model": model, "kind": "best"}
    ) as rl:
        rl.params({f"best.{k}": v for k, v in best.items()})
        rl.metrics({"pareto_size": float(len(study.best_trials))})
    logger.info("%s best params: %s", model, best)
    return study, best
