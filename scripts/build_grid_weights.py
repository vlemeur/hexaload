"""Generate population weights per Open-Meteo grid point and embed them.

NETWORK SCRIPT — run it yourself, once (or when INSEE populations update):
    uv run python scripts/build_grid_weights.py

Source: geo.api.gouv.fr `communes` (fields code, nom, population, centre),
paginated (100/page, ~35k communes — throttled, a few minutes, cached).
Each 1° grid point gets the summed population of communes within 0.7°
(nearest-commune fallback for sea points); weights are normalized to sum 1
and embedded at src/hexaload/features/assets/grid_weights.csv (versioned in
git: the backtest must run offline). The raw API pages are cached under
data/cache/grid-weights/. The matching logic (network-free) lives in
hexaload.features.grid_weights and is unit-tested offline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from hexaload.data.fetch import HttpClient, france_grid
from hexaload.features.grid_weights import assign_grid_weights
from hexaload.features.weather import GRID_WEIGHTS_CSV

logger = logging.getLogger("hexaload.grid_weights")

GEO_API = "https://geo.api.gouv.fr"
PAGE_SIZE = 100
DEFAULT_CACHE = Path("data/cache/grid-weights")


def _communes_path(page: int) -> str:
    return (
        f"communes?fields=code,nom,population,centre&format=json"
        f"&page={page}&page_size={PAGE_SIZE}"
    )


def fetch_communes(
    client: HttpClient, cache_dir: Path, *, refresh: bool = False
) -> list[dict]:
    """Fetch all communes, cached as one JSON in ``cache_dir``.

    geo.api.gouv.fr currently ignores ``page_size`` and returns ALL communes
    on the first page, so only follow pagination when a page comes back
    exactly full; dedupe by code regardless (guards against a non-paginating
    API repeating the same page forever).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "communes.json"
    if raw_path.exists() and not refresh:
        logger.info("cache hit: %s", raw_path)
        return json.loads(raw_path.read_text())
    page = 1
    payload = client.get_json(_communes_path(page), {})
    all_rows: list[dict] = list(payload)
    logger.info("page 1: %d communes", len(payload))
    while len(payload) == PAGE_SIZE and page < 1000:
        page += 1
        payload = client.get_json(_communes_path(page), {})
        if not payload or payload == all_rows[-PAGE_SIZE:]:
            break  # API not paginating: same page repeated
        logger.info(
            "page %d: %d communes (total %d)",
            page,
            len(payload),
            len(all_rows) + len(payload),
        )
        all_rows.extend(payload)
    seen: set = set()
    unique: list[dict] = []
    for row in all_rows:
        if row.get("code") not in seen:
            seen.add(row.get("code"))
            unique.append(row)
    raw_path.write_text(json.dumps(unique, ensure_ascii=False))
    logger.info("fetched %d unique communes", len(unique))
    return unique


def _payload_to_frame(payload: list[dict]) -> pd.DataFrame:
    rows = []
    for c in payload:
        coords = (c.get("centre") or {}).get("coordinates") or [None, None]
        rows.append(
            {
                "code": c.get("code"),
                "nom": c.get("nom"),
                "population": c.get("population"),
                "centre_lon": coords[0],
                "centre_lat": coords[1],
            }
        )
    df = pd.DataFrame(rows)
    if df["population"].isna().all():
        raise RuntimeError("geo.api.gouv.fr returned no population field")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=GRID_WEIGHTS_CSV)
    parser.add_argument("--refresh", action="store_true", help="re-fetch the API")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    client = HttpClient(base_url=GEO_API)
    payload = fetch_communes(client, args.cache_dir, refresh=args.refresh)
    weights = assign_grid_weights(france_grid(), _payload_to_frame(payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(args.out, index=False)
    top = (
        weights.assign(
            latlon=weights["latitude"].astype(str)
            + ","
            + weights["longitude"].astype(str)
        )
        .sort_values("weight", ascending=False)
        .head(3)[["latlon", "weight"]]
        .to_dict("records")
    )
    logger.info(
        "embedded %d grid points into %s (top: %s)", len(weights), args.out, top
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
