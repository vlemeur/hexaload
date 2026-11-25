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
    default_params,
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
    # full-geometry folds (backtest geometry) -> hold-out stays honest
    splits_full = walk_forward_splits(
        y,
        start=cfg.start,
        end=cfg.end,
        horizon=cfg.horizon,
        window_length=cfg.window_length,
        step_length=cfg.step_length,
    )
    # shorter-window folds -> cheaper tuning trials
    splits_tune = walk_forward_splits(
        y,
        start=cfg.start,
        end=cfg.end,
        horizon=cfg.horizon,
        window_length=cfg.tuning_window_length,
        step_length=cfg.step_length,
    )
    logger.info(
        "%d full folds (window=%dh) | %d tuning folds (window=%dh, step=%dh)",
        len(splits_full),
        cfg.window_length,
        len(splits_tune),
        cfg.tuning_window_length,
        cfg.step_length,
    )
    tune = select_folds(splits_tune, cfg)[0]
    holdout = select_folds(splits_full, cfg)[1]
    splits = splits_full
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
    on_fold: Any = None,
) -> list[dict]:
    """One record (mase/crps) per fold for a model instantiated with ``params``.

    ``on_fold(i, record)`` is called after each fold (used to log per-fold
    metrics to MLflow with step=i).
    """
    progress = progress or NullProgress()
    records: list[dict] = []
    for split in splits:
        progress.context(fold=str(split.test_start.date()))
        try:
            records.append(_score_one_fold(model, params, split, data, cfg))
        except (
            Exception
        ) as exc:  # a fold that fails (e.g. MPS OOM) must not kill the run
            logger.warning(
                "fold %s: model %s failed (%s) — skipped",
                split.test_start.date(),
                model,
                exc,
            )
            continue
        if on_fold is not None:
            on_fold(len(records) - 1, records[-1])
    return records


def _score_one_fold(
    model: str,
    params: dict[str, Any],
    split: Split,
    data: TuningData,
    cfg: TuningConfig,
) -> dict:
    y_train = data.y.loc[split.train]
    y_test = data.y.loc[split.test]
    X_fold = (
        fold_features(data.X, data.manifest, split)
        if data.X is not None and data.manifest is not None
        else None
    )
    if model in TORCH_MODELS:
        # same epoch cap as during tuning (the objective injects it too): the
        # hold-out then costs the same per fold as a tuning trial. The full
        # 36-fold backtest (backtest/run.py) is NOT capped — full budget there.
        params = {**params, "max_epochs": cfg.max_epochs_torch}
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
        X_tr = X_fold.loc[split.train][data.exo_cols] if X_fold is not None else None
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
    return rec


def make_epoch_hook(
    progress: BaseProgress,
    best_by_trial: dict[int, float],
    *,
    trial_number: int,
    warmup: int,
    max_epochs: int,
    n_startup: int = 3,
    factor: float = 2.0,
):
    """Per-epoch hook for torch models during tuning.

    Multi-objective Optuna does NOT support ``trial.report`` — so pruning is
    done manually here, MedianPruner-style: track each trial's best val_loss in
    the shared ``best_by_trial`` dict; once at least ``n_startup`` trials have a
    best value and ``warmup`` epochs are done, prune the current trial if its
    val_loss exceeds ``factor`` x the MEDIAN of the per-trial bests (the median,
    not the min — comparing to the absolute best prunes everything).
    """

    def on_epoch_end(epoch: int, val_loss: float) -> None:
        progress.context(epoch=f"{epoch}/{max_epochs}", val=f"{val_loss:.4f}")
        cur = best_by_trial.get(trial_number, float("inf"))
        if val_loss < cur:
            best_by_trial[trial_number] = val_loss
        if epoch < warmup or len(best_by_trial) < n_startup:
            return
        threshold = factor * float(np.median(list(best_by_trial.values())))
        if val_loss > threshold:
            raise optuna.TrialPruned()

    return on_epoch_end


def _objective(
    model: str,
    cfg: TuningConfig,
    data: TuningData,
    progress: BaseProgress,
    best_by_trial: dict[int, float],
    run_factory: Any,
    deadline: float | None = None,
):
    """Optuna objective: mean MASE and mean CRPS over the tuning folds.

    One MLflow run is opened PER TRIAL (via ``run_factory``) so that per-fold
    metrics and torch val_loss-per-epoch are logged with steps, like onomatron.
    ``deadline`` (perf_counter timestamp) is checked after every fold and every
    epoch: a trial that exceeds the wall-clock budget is pruned mid-flight.
    """

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        progress.context(model=model, trial=f"{trial.number}")

        def check_deadline() -> None:
            if deadline is not None and time.perf_counter() > deadline:
                raise optuna.TrialPruned("wall-clock budget exhausted")

        params = suggest_params(model, trial)
        with run_factory(trial) as run:
            run.params({**params, "model": model, "seed": cfg.seed + trial.number})
            try:
                if model in TORCH_MODELS:
                    params.update(
                        max_epochs=cfg.max_epochs_torch,
                        device=cfg.device,
                        seed=cfg.seed + trial.number,
                        # GPU-friendly defaults during tuning: prefetch batches
                        # on CPU workers + bfloat16 autocast on MPS
                        num_workers=4,
                        use_amp=True,
                    )
                    prune_hook = make_epoch_hook(
                        progress,
                        best_by_trial,
                        trial_number=trial.number,
                        warmup=cfg.prune_warmup_epochs,
                        max_epochs=cfg.max_epochs_torch,
                    )

                    def on_epoch_end(epoch: int, val_loss: float) -> None:
                        check_deadline()
                        run.metric("val_loss", val_loss, step=epoch)
                        prune_hook(epoch, val_loss)

                    params["on_epoch_end"] = on_epoch_end

                def on_fold(i: int, rec: dict) -> None:
                    check_deadline()
                    m = {"mase": rec["mase"]}
                    if rec.get("crps") is not None:
                        m["crps"] = rec["crps"]
                    for k, v in m.items():
                        run.metric(f"{k}_fold{i}", v, step=i)

                records = score_folds(
                    model,
                    params,
                    data.tune_splits,
                    data,
                    cfg,
                    progress,
                    on_fold=on_fold,
                )
                mase_mean = float(np.mean([r["mase"] for r in records]))
                crps_vals = [r["crps"] for r in records if r["crps"] is not None]
                crps_mean = float(np.mean(crps_vals)) if crps_vals else PENALTY
                run.metrics({"mase_mean": mase_mean, "crps_mean": crps_mean})
                run.tag("state", "complete")
                return mase_mean, crps_mean
            except optuna.TrialPruned:
                run.tag("state", "pruned")
                raise
            except Exception:
                run.tag("state", "failed")
                raise

    return objective


def pick_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Lowest MASE on the Pareto front among trials within 5% of the best CRPS."""
    front = study.best_trials  # Pareto front for multi-objective studies
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not front:
        if not complete:
            raise RuntimeError("study has no completed trial")
        front = [min(complete, key=lambda t: t.values[0])]  # type: ignore[union-attr]
    min_crps = min(t.values[1] for t in front)  # type: ignore[index]
    cands = [t for t in front if t.values[1] <= min_crps * 1.05]  # type: ignore[index]
    return min(cands, key=lambda t: t.values[0])  # type: ignore[index]


def pick_best_params(study: optuna.Study) -> dict[str, Any]:
    return dict(pick_best_trial(study).params)


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
        # no pruner: multi-objective Optuna does not support trial.report,
        # pruning is done manually in make_epoch_hook for torch models
        load_if_exists=True,
    )
    n_before = len(study.trials)
    # shared across trials: trial_number -> best val_loss (manual median pruning)
    best_by_trial: dict[int, float] = {
        t.number: float("inf") for t in study.trials if t.number is not None
    }
    t0 = time.perf_counter()
    deadline = t0 + cfg.budget_minutes * 60  # also enforced INSIDE each trial
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

    def run_factory(trial: optuna.trial.FrozenTrial) -> RunLogger:
        """One MLflow run per trial, opened INSIDE the objective (stepped metrics)."""
        return RunLogger(
            cfg,
            cfg.experiment,
            f"{model}-trial-{trial.number}",
            {"model": model, "trial": str(trial.number), "study": study_name},
        )

    study.optimize(
        _objective(model, cfg, data, progress, best_by_trial, run_factory, deadline),
        n_trials=cfg.n_trials,
        callbacks=[on_trial_done],
        gc_after_trial=True,
        catch=(Exception,),  # a failing trial must not kill the whole campaign
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
    try:
        best_trial = pick_best_trial(study)
        best = dict(best_trial.params)
    except RuntimeError:
        logger.warning(
            "study %s: no completed trial — falling back to constructor defaults",
            study_name,
        )
        best_trial = None
        best = default_params(model)
    with RunLogger(
        cfg, cfg.experiment, f"{model}-best", {"model": model, "kind": "best"}
    ) as rl:
        rl.params({f"best.{k}": v for k, v in best.items()})
        metrics = {"pareto_size": float(len(study.best_trials))}
        if best_trial is not None and best_trial.values is not None:
            # score of the winning trial, so the -best run is self-contained
            metrics["mase_best"] = float(best_trial.values[0])  # type: ignore[index]
            metrics["crps_best"] = float(best_trial.values[1])  # type: ignore[index]
        rl.metrics(metrics)
    logger.info("%s best params: %s", model, best)
    return study, best
