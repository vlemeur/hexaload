"""Fetch the official school-vacations calendar and embed it in the package.

NETWORK SCRIPT — run it yourself, once (or when the official calendar is
updated):
    uv run python scripts/build_school_calendar.py

The raw CSV is cached under data/cache/school-vacations/ (offline replay)
and the validated copy is embedded at
src/hexaload/features/assets/school_vacations_fr.csv (versioned in git: the
backtest must run fully offline). Logic lives in
hexaload.features.school_calendar (tested offline with mocked HTTP).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from hexaload.features.calendar import SCHOOL_VACATIONS_CSV
from hexaload.features.school_calendar import (
    data_gouv_client,
    download_raw_csv,
    embed_school_calendar,
    find_csv_resource_url,
)

DEFAULT_CACHE = Path("data/cache/school-vacations")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true", help="re-download the CSV")
    parser.add_argument("--out", type=Path, default=SCHOOL_VACATIONS_CSV)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    client = data_gouv_client()
    url = find_csv_resource_url(client)
    raw_path = download_raw_csv(
        url, args.cache_dir, client=client, refresh=args.refresh
    )
    embed_school_calendar(pd.read_csv(raw_path), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
