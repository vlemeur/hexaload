"""Final evaluation on the reserved hold-out (folds >= holdout_start).

Touched once per tuning campaign, with the best params found by
:mod:`hexaload.tuning.tune`. Every model gets one MLflow run in the
``holdout_experiment`` so tuning-vs-holdout gaps are visible at a glance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hexaload.progress import BaseProgress, NullProgress
from hexaload.tuning.config import TuningConfig
from hexaload.tuning.tracking import RunLogger, init_mlflow
from hexaload.tuning.tune import TuningData, score_folds

logger = logging.getLogger(__name__)


def run_holdout(
    model: str,
    params: dict[str, Any],
    cfg: TuningConfig,
    data: TuningData,
    progress: BaseProgress | None = None,
) -> list[dict]:
    """Score one model with fixed params on every hold-out fold, once."""
    progress = progress or NullProgress()
    if not data.holdout_splits:
        raise ValueError("no hold-out folds — check holdout_start")
    records = score_folds(model, params, data.holdout_splits, data, cfg, progress)
    mase_mean = float(np.mean([r["mase"] for r in records]))
    crps_vals = [r["crps"] for r in records if r["crps"] is not None]
    crps_mean = float(np.mean(crps_vals)) if crps_vals else float("nan")
    init_mlflow(cfg)
    with RunLogger(
        cfg,
        cfg.holdout_experiment,
        model,
        {"model": model, "n_folds": str(len(records))},
    ) as rl:
        rl.params(params)
        rl.metrics(
            {
                "mase_mean": mase_mean,
                "crps_mean": crps_mean,
                "n_folds": float(len(records)),
            }
        )
    logger.info(
        "hold-out %s: MASE %.4f  CRPS %.1f  over %d folds",
        model,
        mase_mean,
        crps_mean,
        len(records),
    )
    progress.advance(1, model=model, mase=f"{mase_mean:.4f}", crps=f"{crps_mean:.0f}")
    return records
