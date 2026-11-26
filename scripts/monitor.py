"""Daily monitoring: score the published D-1 forecasts + drift (CUSUM).

Reads the on-disk cache (offline — the forecast.yml workflow refreshes it
first) and ``docs/data/forecasts/*.json``, then writes:

- ``docs/data/monitoring/daily.json``   — per-day MAE/MASE per model + RTE,
  CUSUM on the model-vs-RTE gap
- ``docs/data/monitoring/status.json``  — served / fallback / none + degraded
  inputs

The site's /monitoring/ page fetches these client-side (JSON push, no rebuild).

    uv run --locked python scripts/monitor.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hexaload.backtest.load import load_targets
from hexaload.monitoring import compute_status, daily_scores, json_records
from hexaload.progress import setup_logging

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "docs/data/forecasts"
OUT = ROOT / "docs/data/monitoring"

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, default=FORECASTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    y, ref, _wx, source = load_targets()
    df = daily_scores(args.forecasts, y, ref, source=source)
    args.out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        logger.info("no scored days yet")
        daily_path = args.out / "daily.json"
        daily_path.write_text(json.dumps({"days": []}))
    else:
        # NaN -> None via object dtype: a bare NaN would be invalid strict JSON
        daily_path = args.out / "daily.json"
        daily_path.write_text(json.dumps({"days": json_records(df)}, default=str))
        logger.info("scored %d days -> %s", len(df), daily_path)

    # status: latest published forecast + is RTE's D-1 available for it?
    idx_path = args.forecasts / "index.json"
    rte_ok = False
    if idx_path.exists():
        dates = sorted(json.loads(idx_path.read_text()).get("dates", []))
        if dates:
            j = json.loads((args.forecasts / f"{dates[-1]}.json").read_text())
            rte_ok = any(v is not None for v in (j.get("rte_d1") or []))
    status = compute_status(args.forecasts, rte_available=rte_ok)
    status_path = args.out / "status.json"
    status_path.write_text(json.dumps(status, indent=1))
    logger.info("status: %s (%s) -> %s", status["status"], status["date"], status_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
