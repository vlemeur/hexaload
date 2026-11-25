"""Tuning configuration (defaults mirror onomatron's BenchmarkConfig spirit)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TuningConfig:
    """All knobs of a tuning run. Paths are repo-relative; data/ is gitignored."""

    models: list[str] = field(default_factory=list)  # backtest model names
    n_trials: int = 30  # per model (GBM: more, torch: fewer — see scripts/tune.py)
    tune_folds: int = 3  # folds used for tuning, spread over the pre-holdout period
    holdout_start: str = (
        "2026-01-01"  # folds >= this date = final hold-out, touched once
    )
    budget_minutes: float = (
        60.0  # wall-clock budget per model — enforced inside trials, not just between
    )
    seed: int = 0
    device: str | None = None  # None -> auto (MPS on Apple Silicon, CPU in CI)
    max_epochs_torch: int = (
        20  # cap for torch models during tuning (early stop does the rest)
    )
    wrap: bool = True  # evaluate feature models with EWMA+Mondrian wrappers on
    prune_warmup_epochs: int = 8  # manual ASHA pruning kicks in after this many epochs
    # tuning uses a SHORTER window than the final backtest (cheaper trials; the
    # backtest itself always runs the full window_length below)
    tuning_window_length: int = 24 * 180  # 180 days
    storage: str = "data/tuning/optuna.db"  # SQLite, resumable studies
    mlruns: str = "data/mlruns"  # MLflow file store
    out_params: str = "results/tuning/best_params.yaml"  # generated, never hand-edited
    experiment: str = "hexaload-tuning"
    holdout_experiment: str = "hexaload-holdout"
    # backtest geometry shared with run_backtest.py defaults
    start: str = "2023-01-01"
    end: str = "2026-11-24"
    horizon: int = 24
    window_length: int = 24 * 365
    step_length: int = 24 * 30
