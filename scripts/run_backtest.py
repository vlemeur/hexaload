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
import json
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
from hexaload.backtest.walkforward import (
    DEFAULT_END,
    DEFAULT_HORIZON,
    DEFAULT_START,
    walk_forward_splits,
)
from hexaload.data.fetch import stitch_eco2mix
from hexaload.metrics import DEFAULT_SP, crps, mae, mase, rmse, sample_quantiles
from hexaload.models import (
    DEFAULT_QUANTILES,
    HGBFeatures,
    MLPFeatures,
    RidgeFeatures,
    SktimeAutoETS,
    SktimeTheta,
    XGBFeatures,
)
from hexaload.results.table import summarize, write_table

logger = logging.getLogger("hexaload.backtest")

N_SAMPLES = 100  # ensemble size for the empirical CRPS (from sampled quantiles)

DATA = Path("data/processed")
CONS_DEF_PARQUET = DATA / "eco2mix-national-cons-def-hourly.parquet"
TR_PARQUET = DATA / "eco2mix-national-tr-hourly.parquet"
WX_PARQUET = DATA / "openmeteo-icon_eu-hourly-national.parquet"
FEATURES_PARQUET = DATA / "features-latest.parquet"
FEATURES_MANIFEST = DATA / "features-latest.manifest.json"


def load_features() -> tuple[pd.DataFrame | None, dict | None]:
    """Materialized feature table + manifest (scripts/build_features.py)."""
    if not FEATURES_PARQUET.exists():
        logger.warning(
            "no feature table (%s missing); run scripts/build_features.py — "
            "feature-based models are skipped",
            FEATURES_PARQUET,
        )
        return None, None
    X = pd.read_parquet(FEATURES_PARQUET)
    manifest = json.loads(FEATURES_MANIFEST.read_text())["features"]
    logger.info(
        "features: %d cols x %d rows (%d target-reading)",
        X.shape[1],
        len(X),
        sum(1 for m in manifest.values() if m.get("reads_target")),
    )
    return X, manifest


def load_targets() -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Stitched y/ref (cons-def + real-time tail), weather, and the source mask."""
    cons_def = pd.read_parquet(CONS_DEF_PARQUET)
    tr = pd.read_parquet(TR_PARQUET) if TR_PARQUET.exists() else pd.DataFrame()
    if tr.empty:
        logger.warning(
            "no real-time tail (%s missing); using cons-def only", TR_PARQUET
        )
        eco = cons_def.copy()
        eco["source"] = "cons-def"
    else:
        eco = stitch_eco2mix(cons_def, tr)
        boundary = eco.index[eco["source"] == "tr"][0]
        logger.info(
            "stitched eco2mix: cons-def -> %s, tr from %s", eco.index.max(), boundary
        )
    y = eco["consommation"]
    ref = eco["prevision_j1"]
    source = eco["source"]
    # Exogenous archive forecasts (leakage-safe by construction); not used by
    # the naive baselines yet, loaded to assert the pipeline is complete.
    wx = pd.read_parquet(WX_PARQUET)
    return y, ref, wx, source


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
        if X is not None and manifest is not None:
            X_fold = fold_features(X, manifest, split)
            for name, model_cls in (
                ("ridge_features", RidgeFeatures),
                ("hgb_features", HGBFeatures),
                ("xgb_features", XGBFeatures),
                ("mlp_features", MLPFeatures),
            ):
                if name in skip:
                    continue
                try:
                    model = model_cls().fit(y_train, X_fold.loc[split.train])
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
        for name, model_cls in (
            ("sktime_autoets", SktimeAutoETS),
            ("sktime_theta", SktimeTheta),
        ):
            if name in skip:
                continue
            try:
                model = model_cls().fit(y_train)
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
