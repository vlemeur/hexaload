"""Offline walk-forward backtest: naive baselines + RTE's prevision_j1.

Loads the processed parquets, runs walk-forward folds (sktime splitters),
scores each model (MAE/RMSE/MASE; CRPS is reserved for probabilistic models)
and writes the aggregated table under results/. Run from the repo root:
    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --expanding --step-length 168
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hexaload.backtest.baselines import (
    predict_naive,
    predict_rte_prevision_j1,
    predict_seasonal_naive,
)
from hexaload.backtest.walkforward import (
    DEFAULT_END,
    DEFAULT_HORIZON,
    DEFAULT_START,
    walk_forward_splits,
)
from hexaload.data.fetch import stitch_eco2mix
from hexaload.metrics import DEFAULT_SP, mae, mase, rmse
from hexaload.results.table import summarize, write_table

logger = logging.getLogger("hexaload.backtest")

DATA = Path("data/processed")
CONS_DEF_PARQUET = DATA / "eco2mix-national-cons-def-hourly.parquet"
TR_PARQUET = DATA / "eco2mix-national-tr-hourly.parquet"
WX_PARQUET = DATA / "openmeteo-icon_eu-hourly-national.parquet"


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
        preds = {
            "naive": predict_naive(y_train, split.test),
            "seasonal_naive_24": predict_seasonal_naive(
                y, split.train, split.test, sp=24
            ),
            "seasonal_naive_168": predict_seasonal_naive(
                y, split.train, split.test, sp=168
            ),
            "rte_prevision_j1": predict_rte_prevision_j1(ref, split.test),
        }
        for model, pred in preds.items():
            records.append(
                {
                    "fold": str(split.test_start.date()),
                    "model": model,
                    "mae": mae(y_test, pred),
                    "rmse": rmse(y_test, pred),
                    "mase": mase(y_test, pred, y_train=y_train, sp=DEFAULT_SP),
                }
            )
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
