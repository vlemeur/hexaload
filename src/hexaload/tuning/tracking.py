"""MLflow local tracking (SQLite backend), à la onomatron's RunLogger.

Everything stays on disk under ``data/mlruns/mlflow.db`` (data/ is gitignored) —
no server, no network. Serve the UI locally with:
    uv run mlflow ui --backend-store-uri sqlite://$PWD/data/mlruns/mlflow.db --port 5000
"""

from __future__ import annotations

import logging
import resource
import time
from pathlib import Path
from typing import Any

import mlflow

from .config import TuningConfig

logger = logging.getLogger(__name__)


_last_uri: str | None = None  # log the tracking URI only once per process


def init_mlflow(cfg: TuningConfig) -> str:
    """Point MLflow at the local SQLite store; return the tracking URI (idempotent)."""
    global _last_uri
    db = Path(cfg.mlruns).resolve() / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.experiment)
    if uri != _last_uri:
        logger.info("mlflow tracking uri: %s (experiment %s)", uri, cfg.experiment)
        _last_uri = uri
    return uri


def peak_mb() -> float:
    """Peak RSS of this process in MB (macOS: ru_maxrss is bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class RunLogger:
    """Context manager = one MLflow run. Logs params, metrics, wall time, peak MB."""

    def __init__(
        self, cfg: TuningConfig, experiment: str, run_name: str, tags: dict[str, str]
    ) -> None:
        self._cfg = cfg
        self._experiment = experiment
        self._run_name = run_name
        self._tags = tags
        self._t0 = 0.0

    def __enter__(self) -> RunLogger:
        Path(self._cfg.mlruns).mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self._experiment)
        self._t0 = time.perf_counter()
        mlflow.start_run(run_name=self._run_name, tags=self._tags)
        return self

    def params(self, d: dict[str, Any]) -> None:
        flat = {
            k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in d.items()
        }
        mlflow.log_params(flat)

    def metrics(self, d: dict[str, float]) -> None:
        mlflow.log_metrics({k: float(v) for k, v in d.items()})

    def metric(self, key: str, value: float, step: int = 0) -> None:
        """One stepped metric point (e.g. val_loss per epoch, mase per fold)."""
        mlflow.log_metric(key, float(value), step=step)

    def tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.metrics(
            {
                "wall_time_s": time.perf_counter() - self._t0,
                "peak_mb": peak_mb(),
            }
        )
        mlflow.end_run()
