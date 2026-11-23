"""Fetch open data into the on-disk cache, then validate it (offline replay).

The eco2mix national series is fetched across the cons-def -> temps-réel feed
switch (2026-08-31) automatically: one command covers any [start, end) range.
The real-time feed keeps only ~85 days, so re-run this regularly to archive
the tail. VPN OFF when fetching ODRE/RTE: the ODRE WAF resets TLS handshakes
from some VPN exit IPs. Examples:
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

import pandas as pd

from hexaload.data.fetch import (
    CONS_DEF_DATASET,
    DEFAULT_DATASET,
    TR_DATASET,
    fetch_dataset,
    fetch_eco2mix,
    fetch_openmeteo_dataset,
    read_cache,
    stitch_eco2mix,
)
from hexaload.data.validate import (
    DEFAULT_NATURES,
    NATURE_TR,
    build_hourly_dataset,
    build_weather_dataset,
)

DEFAULT_START = {
    # selecting 2023 as start date for both consumption and meteo data
    # so that both are consistent
    "eco2mix": "2023-01-01",
    "openmeteo": "2023-01-01",
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
        if source == "eco2mix":
            # default: fetch across the cons-def -> tr switch transparently;
            # an explicit --dataset fetches that dataset only.
            datasets = (
                [CONS_DEF_DATASET, TR_DATASET]
                if args.dataset == DEFAULT_DATASET
                else [args.dataset]
            )
            if not args.offline:
                if args.dataset == DEFAULT_DATASET:
                    files = fetch_eco2mix(
                        start, end, args.cache_dir, refresh=args.refresh, progress=True
                    )
                else:
                    files = fetch_dataset(
                        args.dataset,
                        start,
                        end,
                        args.cache_dir,
                        refresh=args.refresh,
                        progress=True,
                    )
                logging.info("%s: cache covers %d monthly files", source, len(files))
            if not args.skip_validate:
                for dataset in datasets:
                    _validate_eco2mix(args, dataset)
                _write_stitched(args)
        else:
            if not args.offline:
                files = fetch_openmeteo_dataset(
                    start,
                    end,
                    args.cache_dir,
                    refresh=args.refresh,
                    progress=True,
                )
                logging.info("%s: cache covers %d monthly files", source, len(files))
            if not args.skip_validate:
                dataset = "openmeteo-icon_eu"
                raw = read_cache(dataset, args.cache_dir)
                logging.info("%s: %d raw records", source, len(raw))
                hourly, report = build_weather_dataset(raw)
                out = args.processed_dir / f"{dataset}-hourly-national.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                hourly.to_parquet(out)
                logging.info(
                    "wrote %s (%d rows); %s", out, len(hourly), report.summary()
                )
    return 0


def _validate_eco2mix(args: argparse.Namespace, dataset: str) -> None:
    """Validate one cached eco2mix dataset into its hourly processed parquet."""
    cache = args.cache_dir / dataset
    if not any(cache.glob("*.parquet")):
        logging.info("eco2mix: no cache for %s, skipping validation", dataset)
        return
    raw = read_cache(dataset, args.cache_dir)
    logging.info("eco2mix %s: %d raw records", dataset, len(raw))
    # "*-tr" (temps réel) datasets only carry the real-time nature.
    natures = (NATURE_TR,) if dataset.endswith("-tr") else DEFAULT_NATURES
    hourly, report = build_hourly_dataset(raw, natures=natures)
    out = args.processed_dir / f"{dataset}-hourly.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out)
    logging.info("wrote %s (%d rows); %s", out, len(hourly), report.summary())


def _write_stitched(args: argparse.Namespace) -> None:
    """Write the stitched national hourly parquet (cons-def + tr tail)."""
    cons_pq = args.processed_dir / f"{CONS_DEF_DATASET}-hourly.parquet"
    tr_pq = args.processed_dir / f"{TR_DATASET}-hourly.parquet"
    if not cons_pq.exists():
        return
    cons_def = pd.read_parquet(cons_pq)
    tr = pd.read_parquet(tr_pq) if tr_pq.exists() else cons_def.iloc[:0]
    out = args.processed_dir / "eco2mix-national-hourly.parquet"
    stitch_eco2mix(cons_def, tr).to_parquet(out)
    logging.info("wrote stitched %s", out)


if __name__ == "__main__":
    sys.exit(main())
