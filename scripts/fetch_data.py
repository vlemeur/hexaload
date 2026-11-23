"""Fetch eco2mix data into the on-disk cache, then validate it (offline replay).

VPN OFF when fetching: the ODRE WAF resets TLS handshakes from some VPN exit
IPs. Examples:
    uv run python scripts/fetch_data.py --from 2012-02-22 --to 2026-12-01
    uv run python scripts/fetch_data.py --offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from hexaload.data.fetch import DEFAULT_DATASET, fetch_dataset, read_cache
from hexaload.data.validate import build_hourly_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--from",
        dest="start",
        default="2012-02-22",
        help="range start, UTC (default: first eco2mix record)",
    )
    parser.add_argument("--to", dest="end", default=None, help="range end, UTC")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--refresh", action="store_true", help="re-fetch months already cached"
    )
    parser.add_argument(
        "--offline", action="store_true", help="no network; replay the cache only"
    )
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=UTC)
        if args.end
        else datetime.now(UTC)
    )
    if not args.offline:
        files = fetch_dataset(
            args.dataset,
            start,
            end,
            args.cache_dir,
            refresh=args.refresh,
            progress=True,
        )
        logging.info("cache covers %d monthly files", len(files))
    raw = read_cache(args.dataset, args.cache_dir)
    logging.info("raw records: %d", len(raw))
    if args.skip_validate:
        return 0
    hourly, report = build_hourly_dataset(raw)
    out = args.processed_dir / f"{args.dataset}-hourly.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out)
    logging.info("wrote %s (%d hourly rows); %s", out, len(hourly), report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
