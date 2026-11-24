"""Build the full feature table offline from the on-disk cache.

    uv run python scripts/build_features.py

Writes data/processed/features-<digest>.parquet + manifest (delays per
column, sha256 of every cache input, horizon) and refreshes the stable
features-latest.* aliases that the backtest reads. Logic lives in
hexaload.features.build (unit-tested offline).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hexaload.features.build import DEFAULT_HORIZON, build

DEFAULT_CACHE = Path("data/cache")
DEFAULT_OUT_DIR = Path("data/processed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    build(args.cache_dir, args.out_dir, horizon=args.horizon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
