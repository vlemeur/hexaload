"""Serving status of the daily forecast pipeline.

- ``served``    : a forecast JSON exists for the latest target day and its
                  exogenous inputs were complete.
- ``fallback``  : a forecast exists but was produced with degraded inputs
                  (e.g. live weather missing → archive-only) or RTE's D-1
                  forecast is not published yet.
- ``none``      : no forecast published at all.
"""

from __future__ import annotations

import json
from pathlib import Path


def compute_status(
    forecast_dir: Path | str,
    *,
    rte_available: bool,
    live_weather_ok: bool = True,
) -> dict:
    """Derive the pipeline status from the published forecast files."""
    forecast_dir = Path(forecast_dir)
    idx_path = forecast_dir / "index.json"
    if not idx_path.exists():
        return {"status": "none", "degraded_inputs": [], "date": None}
    dates = sorted(json.loads(idx_path.read_text()).get("dates", []))
    if not dates:
        return {"status": "none", "degraded_inputs": [], "date": None}
    latest = dates[-1]
    degraded: list[str] = []
    if not rte_available:
        degraded.append("rte_prevision_j1 not published yet for the target day")
    if not live_weather_ok:
        degraded.append("live weather unavailable (archive-only forecast)")
    return {
        "status": "fallback" if degraded else "served",
        "degraded_inputs": degraded,
        "date": latest,
    }
