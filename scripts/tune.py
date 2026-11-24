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
        choices=("tune", "eval", "all"),
        default="all",
        help="tune: Optuna only | eval: hold-out from best_params.yaml only | all",
    )
    parser.add_argument("--device", default=None, help="cpu|mps|cuda (default: auto)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    cfg = TuningConfig(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        n_trials=args.trials,
        tune_folds=args.folds,
        budget_minutes=args.budget_minutes,
        device=args.device,
    )
    init_mlflow(cfg)
    data = load_tuning_data(cfg)

    if args.mode in ("tune", "all"):
        best_all = load_best_params(cfg.out_params)
        for model in cfg.models:
            _study, best = run_study(model, cfg, data)
            best_all[model] = best
        save_best_params(best_all, cfg.out_params)

    if args.mode in ("eval", "all"):
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
                    model, best_all.get(model, default_params(model)), cfg, data
                )
            )
        table = summarize(records)
        out = Path("results/tuning/holdout.parquet")
        write_table(table, out, out.with_suffix(".md"))
        logger.info("wrote %s and %s", out, out.with_suffix(".md"))
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
