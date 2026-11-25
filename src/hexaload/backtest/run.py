"""Offline walk-forward backtest engine (library side of scripts/run_backtest.py).

Baselines + RTE's prevision_j1 (absolute reference) + all model families,
scored fold by fold: MAE/RMSE/MASE, plus CRPS from sampled quantiles for every
model exposing ``predict_quantiles`` (baselines get split-conformal intervals
from train-fold residuals). The aggregated table is written under results/.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hexaload.backtest.baselines import (
    naive_residuals,
    predict_naive,
    predict_rte_prevision_j1,
    predict_seasonal_naive,
    residual_quantiles,
    seasonal_residuals,
)
from hexaload.backtest.features import fold_features
from hexaload.backtest.load import exogenous_columns, load_features, load_targets
from hexaload.backtest.walkforward import (
    DEFAULT_END,
    DEFAULT_HORIZON,
    DEFAULT_START,
    walk_forward_splits,
)
from hexaload.metrics import DEFAULT_SP, crps, mae, mase, rmse, sample_quantiles
from hexaload.models import DEFAULT_QUANTILES
from hexaload.progress import Progress
from hexaload.results.table import summarize, write_table
from hexaload.tuning.params import load_best_params
from hexaload.tuning.spaces import (
    FEATURE_MODELS,
    TSFM_MODELS,
    UNIVARIATE_MODELS,
    VARIANT_MODELS,
    build_model,
    build_variant,
)

logger = logging.getLogger("hexaload.backtest")

N_SAMPLES = 100  # ensemble size for the empirical CRPS (from sampled quantiles)


@dataclass
class BacktestConfig:
    """One backtest run. ``params``: optional best-params YAML (tuned models)."""

    start: str = DEFAULT_START
    end: str = DEFAULT_END
    horizon: int = DEFAULT_HORIZON
    window_length: int = 24 * 365
    step_length: int = 24 * 30
    expanding: bool = False
    skip: str = ""
    params: str | None = None
    drop_features: str = ""  # comma-separated columns to remove (e.g. no-RTE variant)
    out: str = "results/backtest.parquet"


def run(cfg: BacktestConfig) -> pd.DataFrame:
    y, ref, wx, source = load_targets()
    X, manifest = load_features()
    logger.info(
        "target: %d hourly steps %s -> %s (%s) | weather: %d x %d",
        len(y),
        y.index.min(),
        y.index.max(),
        "/".join(source.unique()),
        *wx.shape,
    )
    splits = walk_forward_splits(
        y,
        start=cfg.start,
        end=cfg.end,
        horizon=cfg.horizon,
        window_length=cfg.window_length,
        step_length=cfg.step_length,
        expanding=cfg.expanding,
    )
    logger.info(
        "%d walk-forward folds (window=%dh, step=%dh, horizon=%dh, %s)",
        len(splits),
        cfg.window_length,
        cfg.step_length,
        cfg.horizon,
        "expanding" if cfg.expanding else "sliding",
    )
    model_params: dict[str, dict] = {}
    if cfg.params:
        model_params = load_best_params(cfg.params)
        logger.info("tuned params from %s (%d models)", cfg.params, len(model_params))
    skip = {s.strip() for s in cfg.skip.split(",") if s.strip()}
    base_names = (
        "naive",
        "seasonal_naive_24",
        "seasonal_naive_168",
        "rte_prevision_j1",
    )
    n_expected = len([n for n in base_names if n not in skip])
    if X is not None:
        n_expected += len([n for n in FEATURE_MODELS if n not in skip])
        n_expected += len([n for n in VARIANT_MODELS if n not in skip])
    n_expected += len([n for n in UNIVARIATE_MODELS if n not in skip])
    n_expected += len([n for n in TSFM_MODELS if n not in skip])
    prog = Progress(
        total=len(splits) * max(1, n_expected), desc="walk-forward", unit="model·fold"
    )
    records: list[dict] = []
    for split in splits:
        prog.context(fold=str(split.test_start.date()))
        y_train = y.loc[split.train]
        y_test = y.loc[split.test]
        preds: dict[str, pd.Series] = {}
        qpreds: dict[str, pd.DataFrame] = {}
        # D-1-only references (RTE prevision_j1, seasonal-naive sp=24) are only
        # defined on the first 24 h of a longer (e.g. D+7) test block
        d1_test = split.test[: min(24, len(split.test))]
        baseline_specs = {
            "naive": (predict_naive(y_train, split.test), naive_residuals(y_train)),
            "seasonal_naive_24": (
                predict_seasonal_naive(y, split.train, d1_test, sp=24),
                seasonal_residuals(y_train, sp=24),
            ),
            "seasonal_naive_168": (
                predict_seasonal_naive(y, split.train, split.test, sp=168),
                seasonal_residuals(y_train, sp=168),
            ),
            "rte_prevision_j1": (
                predict_rte_prevision_j1(ref, d1_test),
                (y_train - ref.reindex(y_train.index)).dropna(),
            ),
        }
        for name, (pred, resid) in baseline_specs.items():
            if name in skip:
                continue
            preds[name] = pred
            # split-conformal intervals from train-fold residuals (as-of)
            qpreds[name] = residual_quantiles(pred, resid, DEFAULT_QUANTILES)
        X_fold = (
            fold_features(X, manifest, split)
            if X is not None and manifest is not None
            else None
        )
        drop = {c.strip() for c in cfg.drop_features.split(",") if c.strip()}
        if X_fold is not None and drop:
            X_fold = X_fold.drop(columns=[c for c in drop if c in X_fold.columns])
        if X_fold is not None:
            for name in FEATURE_MODELS:
                if name in skip:
                    continue
                prog.context(model=name)
                try:
                    model = build_model(name, model_params.get(name)).fit(
                        y_train, X_fold.loc[split.train]
                    )
                    preds[name] = model.predict(split.test, X_fold.loc[split.test])
                except Exception as exc:
                    logger.warning(
                        "fold %s: model %s failed (%s) — no score for this fold",
                        split.test_start.date(),
                        name,
                        exc,
                    )
                    continue
                # point-only models raise NotImplementedError -> no CRPS
                with suppress(NotImplementedError):
                    qpreds[name] = model.predict_quantiles(
                        split.test, X_fold.loc[split.test], DEFAULT_QUANTILES
                    )
        for name in VARIANT_MODELS:
            if name in skip:
                continue
            prog.context(model=name)
            try:
                model = build_variant(name, model_params).fit(
                    y_train, X_fold.loc[split.train]
                )
                preds[name] = model.predict(split.test, X_fold.loc[split.test])
            except Exception as exc:
                logger.warning(
                    "fold %s: model %s failed (%s) — no score for this fold",
                    split.test_start.date(),
                    name,
                    exc,
                )
                continue
            with suppress(NotImplementedError):
                qpreds[name] = model.predict_quantiles(
                    split.test, X_fold.loc[split.test], DEFAULT_QUANTILES
                )
        for name in UNIVARIATE_MODELS:
            if name in skip:
                continue
            prog.context(model=name)
            try:
                model = build_model(name, model_params.get(name)).fit(y_train)
                preds[name] = model.predict(split.test)
            except Exception as exc:
                logger.warning(
                    "fold %s: model %s failed (%s) — no score for this fold",
                    split.test_start.date(),
                    name,
                    exc,
                )
                continue
            # point-only models raise NotImplementedError -> no CRPS
            with suppress(NotImplementedError):
                qpreds[name] = model.predict_quantiles(
                    split.test, quantiles=DEFAULT_QUANTILES
                )
        # zero-shot TSFMs: exogenous columns only as known-future covariates
        exo_cols = exogenous_columns(manifest)
        for name in TSFM_MODELS:
            if name in skip:
                continue
            prog.context(model=name)
            try:
                X_tr = X_fold.loc[split.train][exo_cols] if X_fold is not None else None
                X_te = X_fold.loc[split.test][exo_cols] if X_fold is not None else None
                model = build_model(name, model_params.get(name)).fit(y_train, X_tr)
                preds[name] = model.predict(split.test, X_te)
            except Exception as exc:
                logger.warning(
                    "fold %s: model %s failed (%s) — no score for this fold",
                    split.test_start.date(),
                    name,
                    exc,
                )
                continue
            with suppress(NotImplementedError):
                qpreds[name] = model.predict_quantiles(
                    split.test, X_te, DEFAULT_QUANTILES
                )
        for model, pred in preds.items():
            # D-1-only references (RTE prevision_j1, seasonal-naive sp=24):
            # with a longer horizon (D+7), score them on the first 24 h only
            if (
                model in ("rte_prevision_j1", "seasonal_naive_24")
                and len(split.test) > 24
            ):
                test_eval = split.test[:24]
            else:
                test_eval = split.test
            y_eval = y_test.loc[test_eval]
            pred_eval = pred.reindex(test_eval)
            rec = {
                "fold": str(split.test_start.date()),
                "model": model,
                "mae": mae(y_eval, pred_eval),
                "rmse": rmse(y_eval, pred_eval),
                "mase": mase(y_eval, pred_eval, y_train=y_train, sp=DEFAULT_SP),
            }
            if model in qpreds:
                samples = sample_quantiles(
                    qpreds[model].reindex(test_eval),
                    DEFAULT_QUANTILES,
                    N_SAMPLES,
                    seed=0,
                )
                rec["crps"] = crps(y_eval, samples)
            records.append(rec)
            prog.advance(1, model=model)
    prog.close()
    table = summarize(records)
    out_parquet = Path(cfg.out)
    write_table(table, out_parquet, out_parquet.with_suffix(".md"))
    logger.info("wrote %s and %s", out_parquet, out_parquet.with_suffix(".md"))
    return table
