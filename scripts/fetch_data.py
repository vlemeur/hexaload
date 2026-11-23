"""Fetch open data into the on-disk cache, then validate it (offline replay).

VPN OFF when fetching ODRE/RTE: the ODRE WAF resets TLS handshakes from some
VPN exit IPs. Examples:
    uv run python scripts/fetch_data.py --source all --to 2026-12-01
    uv run python scripts/fetch_data.py --source openmeteo
    uv run python scripts/fetch_data.py --offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from hexaload.data.fetch import (
    DEFAULT_DATASET,
    fetch_dataset,
    fetch_openmeteo_dataset,
    read_cache,
)
from hexaload.data.validate import build_hourly_dataset, build_weather_dataset

DEFAULT_START = {
    "eco2mix": "2012-02-22",  # first eco2mix-national-cons-def record
    "openmeteo": "2016-01-01",  # first archived forecast on Open-Meteo
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["eco2mix", "openmeteo", "all"],
        default="all",
        help="which source to fetch/validate (default: all)",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--from", dest="start", default=None, help="range start, UTC")
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


def _range(args: argparse.Namespace, source: str) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(args.start or DEFAULT_START[source]).replace(
        tzinfo=UTC
    )
    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=UTC)
        if args.end
        else datetime.now(UTC)
    )
    return start, end


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()
    sources = ["eco2mix", "openmeteo"] if args.source == "all" else [args.source]

    for source in sources:
        start, end = _range(args, source)
        if not args.offline:
            if source == "eco2mix":
                files = fetch_dataset(
                    args.dataset,
                    start,
                    end,
                    args.cache_dir,
                    refresh=args.refresh,
                    progress=True,
                )
            else:
                files = fetch_openmeteo_dataset(
                    start,
                    end,
                    args.cache_dir,
                    refresh=args.refresh,
                    progress=True,
                )
            logging.info("%s: cache covers %d monthly files", source, len(files))
        if args.skip_validate:
            continue
        if source == "eco2mix":
            raw = read_cache(args.dataset, args.cache_dir)
            logging.info("%s: %d raw records", source, len(raw))
            hourly, report = build_hourly_dataset(raw)
            out = args.processed_dir / f"{args.dataset}-hourly.parquet"
        else:
            dataset = "openmeteo-icon_eu"
            raw = read_cache(dataset, args.cache_dir)
            logging.info("%s: %d raw records", source, len(raw))
            hourly, report = build_weather_dataset(raw)
            out = args.processed_dir / f"{dataset}-hourly-national.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_parquet(out)
        logging.info("wrote %s (%d rows); %s", out, len(hourly), report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
