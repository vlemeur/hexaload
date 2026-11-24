"""Hyperparameter tuning (Optuna) + experiment tracking (MLflow), local-first.

Inspired by onomatron's benchmark package: persistent SQLite Optuna studies
(resumable), seeded samplers, ASHA-style pruning via per-epoch val-loss reports,
MLflow file tracking, and a strict separation between tuning folds and a final
hold-out (touched once, at evaluation time).
"""

from .config import TuningConfig

__all__ = ["TuningConfig"]
