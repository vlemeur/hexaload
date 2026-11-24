"""Offline walk-forward backtest: baselines + RTE's prevision_j1 + models.

Models: naive/seasonal-naive baselines, RTE's own D-1 forecast (absolute
reference), feature-based models (ridge, HGB) and univariate sktime stats
forecasters (ETS, Theta). Loads the processed parquets,
runs walk-forward folds (sktime splitters), scores each model (MAE/RMSE/MASE,
plus CRPS from sampled quantiles for every model that exposes
``predict_quantiles`` — baselines get split-conformal intervals from train-fold
residuals) and writes the aggregated table under results/. Run from the repo
root:
    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --expanding --step-length 168
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import suppress
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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
from hexaload.results.table import summarize, write_table
from hexaload.tuning.params import load_best_params
from hexaload.tuning.spaces import (
    FEATURE_MODELS,
    TSFM_MODELS,
    UNIVARIATE_MODELS,
    build_model,
)

logger = logging.getLogger("hexaload.backtest")

N_SAMPLES = 100  # ensemble size for the empirical CRPS (from sampled quantiles)


def run(args: argparse.Namespace) -> pd.DataFrame:
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
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        window_length=args.window_length,
        step_length=args.step_length,
        expanding=args.expanding,
    )
    logger.info(
        "%d walk-forward folds (window=%dh, step=%dh, horizon=%dh, %s)",
        len(splits),
        args.window_length,
        args.step_length,
        args.horizon,
        "expanding" if args.expanding else "sliding",
    )
    model_params: dict[str, dict] = {}
    if args.params:
        model_params = load_best_params(args.params)
        logger.info("tuned params from %s (%d models)", args.params, len(model_params))
    records: list[dict] = []
    for split in tqdm(splits, desc="walk-forward", unit="fold"):
        y_train = y.loc[split.train]
        y_test = y.loc[split.test]
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        preds: dict[str, pd.Series] = {}
        qpreds: dict[str, pd.DataFrame] = {}
        baseline_specs = {
            "naive": (predict_naive(y_train, split.test), naive_residuals(y_train)),
            "seasonal_naive_24": (
                predict_seasonal_naive(y, split.train, split.test, sp=24),
                seasonal_residuals(y_train, sp=24),
            ),
            "seasonal_naive_168": (
                predict_seasonal_naive(y, split.train, split.test, sp=168),
                seasonal_residuals(y_train, sp=168),
            ),
            "rte_prevision_j1": (
                predict_rte_prevision_j1(ref, split.test),
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
        if X_fold is not None:
            for name in FEATURE_MODELS:
                if name in skip:
                    continue
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
        for name in UNIVARIATE_MODELS:
            if name in skip:
                continue
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
            rec = {
                "fold": str(split.test_start.date()),
                "model": model,
                "mae": mae(y_test, pred),
                "rmse": rmse(y_test, pred),
                "mase": mase(y_test, pred, y_train=y_train, sp=DEFAULT_SP),
            }
            if model in qpreds:
                samples = sample_quantiles(
                    qpreds[model], DEFAULT_QUANTILES, N_SAMPLES, seed=0
                )
                rec["crps"] = crps(y_test, samples)
            records.append(rec)
    table = summarize(records)
    out_parquet = Path(args.out)
    write_table(table, out_parquet, out_parquet.with_suffix(".md"))
    logger.info("wrote %s and %s", out_parquet, out_parquet.with_suffix(".md"))
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="window start, UTC")
    parser.add_argument("--end", default=DEFAULT_END, help="window end, UTC")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--window-length",
        type=int,
        default=24 * 365,
        help="train window in hourly steps (default: 1 year)",
    )
    parser.add_argument(
        "--step-length",
        type=int,
        default=24 * 30,
        help="steps between folds (default: 30 days)",
    )
    parser.add_argument(
        "--expanding", action="store_true", help="expanding instead of sliding window"
    )
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated model names to skip (e.g. 'sktime_autoarima')",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="YAML of per-model tuned params (e.g. results/tuning/best_params.yaml)",
    )
    parser.add_argument("--out", default="results/backtest.parquet")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    table = run(args)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
