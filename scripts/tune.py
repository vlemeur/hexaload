"""Tune hyperparameters with Optuna, evaluate on the reserved hold-out.

Multi-objective (MASE + CRPS, NSGA-II), persistent SQLite studies, local MLflow
— all resumable and offline (data comes from data/processed):

    just tune ridge_features,hgb_features 10 20        # smoke (~20 min)
    just tune hgb_features,xgb_features 50 120         # GBM campaign
    just tune mlp_features,rnn_features,tcn_features,nhits,patchtst_features 20 240
    just tune timestral_209m,chronos_bolt 6 60
    just tune-eval                                     # 2026 hold-out, once
    just tune-ui                                       # MLflow UI on :5000

Best params land in results/tuning/best_params.yaml, consumed by
run_backtest.py via --params results/tuning/best_params.yaml.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hexaload.progress import Progress, setup_logging
from hexaload.results.table import summarize, write_table
from hexaload.tuning.config import TuningConfig
from hexaload.tuning.evaluate import run_holdout
from hexaload.tuning.params import load_best_params, save_best_params
from hexaload.tuning.spaces import TUNABLE, default_params
from hexaload.tuning.tracking import init_mlflow
from hexaload.tuning.tune import load_tuning_data, run_study

logger = logging.getLogger("hexaload.tuning")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(TUNABLE),
        help="comma-separated backtest model names (default: all tunable)",
    )
    parser.add_argument(
        "--trials", type=int, default=30, help="Optuna trials per model"
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=90.0,
        help="wall-clock budget per model study (study stops between trials)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=6,
        help="number of tuning folds (spread pre-holdout)",
    )
    parser.add_argument(
        "--mode",
        choices=("tune", "eval", "all", "full"),
        default="all",
        help="tune: Optuna only | eval: hold-out only | all: tune+eval | "
        "full: all + backtests (tuned and default) in one command",
    )
    parser.add_argument("--device", default=None, help="cpu|mps|cuda (default: auto)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep optuna/mlflow/data logs (default: quiet, bar stays readable)",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    cfg = TuningConfig(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        n_trials=args.trials,
        tune_folds=args.folds,
        budget_minutes=args.budget_minutes,
        device=args.device,
    )
    init_mlflow(cfg)
    data = load_tuning_data(cfg)

    # one bar for the whole campaign: n_trials per model in tuning + 1 eval per model
    doing_all = args.mode in ("tune", "all", "full")
    n_tune = len(cfg.models) * cfg.n_trials if doing_all else 0
    n_eval = len(cfg.models) if args.mode in ("eval", "all", "full") else 0
    with Progress(total=max(1, n_tune + n_eval), desc=args.mode, unit="run") as prog:
        if doing_all:
            best_all = load_best_params(cfg.out_params)
            for model in cfg.models:
                _study, best = run_study(model, cfg, data, prog)
                best_all[model] = best
            save_best_params(best_all, cfg.out_params)

        if args.mode in ("eval", "all", "full"):
            best_all = load_best_params(cfg.out_params)
            missing = [m for m in cfg.models if m not in best_all]
            if missing:
                logger.warning(
                    "no tuned params in %s for %s — falling back to defaults",
                    cfg.out_params,
                    missing,
                )
            records: list[dict] = []
            for model in cfg.models:
                records.extend(
                    run_holdout(
                        model,
                        best_all.get(model, default_params(model)),
                        cfg,
                        data,
                        prog,
                    )
                )
            table = summarize(records)
            out = Path("results/tuning/holdout.parquet")
            write_table(table, out, out.with_suffix(".md"))
            logger.info("wrote %s and %s", out, out.with_suffix(".md"))
            print(table.to_string(index=False))

    if args.mode == "full":
        # everything evaluated on the same folds, one command: tuned then default
        from hexaload.backtest.run import BacktestConfig
        from hexaload.backtest.run import run as backtest_run

        base = dict(
            start=cfg.start,
            end=cfg.end,
            horizon=cfg.horizon,
            window_length=cfg.window_length,
            step_length=cfg.step_length,
        )
        logger.info("backtest with tuned params -> results/backtest_tuned.parquet")
        backtest_run(
            BacktestConfig(
                **base, params=cfg.out_params, out="results/backtest_tuned.parquet"
            )
        )
        logger.info(
            "backtest with constructor defaults -> results/backtest_default.parquet"
        )
        backtest_run(BacktestConfig(**base, out="results/backtest_default.parquet"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
