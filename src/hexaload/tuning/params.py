"""Best-params YAML I/O — no Optuna/MLflow imports here, so run_backtest.py can
consume ``results/tuning/best_params.yaml`` without the tuning stack installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def save_best_params(all_params: dict[str, dict[str, Any]], path: str | Path) -> None:
    """Write the generated best-params YAML (never hand-edited)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(all_params, sort_keys=True, allow_unicode=True))
    logger.info("wrote %s (%d models)", out, len(all_params))


def load_best_params(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read a best-params YAML; missing file -> {} (constructor defaults)."""
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}
