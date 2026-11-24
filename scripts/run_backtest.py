"""CLI for the offline walk-forward backtest (engine: hexaload.backtest.run).

Evaluates ALL models on the same walk-forward folds in one run (baselines,
RTE's prevision_j1 reference, feature models, univariate stats, TSFMs):

    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --expanding --step-length 168
    uv run python scripts/run_backtest.py --params results/tuning/best_params.yaml \
        --out results/backtest_tuned.parquet
"""

from __future__ import annotations

import argparse
import sys

from hexaload.backtest.run import BacktestConfig, run
from hexaload.backtest.walkforward import DEFAULT_END, DEFAULT_HORIZON, DEFAULT_START
from hexaload.progress import setup_logging


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep third-party/data logs (default: quiet, bar stays readable)",
    )
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    cfg = BacktestConfig(
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        window_length=args.window_length,
        step_length=args.step_length,
        expanding=args.expanding,
        skip=args.skip,
        params=args.params,
        out=args.out,
    )
    table = run(cfg)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
