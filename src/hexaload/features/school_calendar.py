"""Official French school-vacations calendar: locate, download, embed.

Source: data.gouv.fr dataset "vacances-scolaires-par-zones", resource
"vacances.csv" (github.com/AntoineAugusti/vacances-scolaires) — one row per
day, 1990 -> 2030, boolean flags per zone A/B/C plus the vacation name.

Network lives here (and only here); the validated result is embedded as a
versioned CSV inside the package (``features/assets/school_vacations_fr.csv``)
so the backtest runs fully offline. ``scripts/build_school_calendar.py`` is
the thin CLI over this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from hexaload.data.fetch import HttpClient

from .calendar import SCHOOL_VACATIONS_CSV, validate_school_vacations

logger = logging.getLogger(__name__)

DATA_GOUV_API = "https://www.data.gouv.fr/api/1"
SCHOOL_DATASET_PATH = "datasets/vacances-scolaires-par-zones/"
EMBED_START = "2012-01-01"  # the consumption series starts 2012-03


def data_gouv_client() -> HttpClient:
    """Throttled HTTP client bound to the data.gouv.fr API (not ODRE)."""
    return HttpClient(base_url=DATA_GOUV_API)


def find_csv_resource_url(client: HttpClient | None = None) -> str:
    """Locate the 'vacances.csv' resource URL of the data.gouv.fr dataset."""
    client = client or data_gouv_client()
    payload = client.get_json(SCHOOL_DATASET_PATH, {})
    for resource in payload.get("resources", []):
        title = str(resource.get("title", "")).lower()
        fmt = str(resource.get("format", "")).lower()
        if "vacances" in title and "csv" in fmt and resource.get("url"):
            return str(resource["url"])
    raise RuntimeError(f"no vacances CSV resource found in {SCHOOL_DATASET_PATH}")


def download_raw_csv(
    url: str,
    cache_dir: Path,
    *,
    client: HttpClient | None = None,
    refresh: bool = False,
) -> Path:
    """Download the raw CSV into the on-disk cache (offline replay afterwards)."""
    client = client or data_gouv_client()
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "vacances.csv"
    manifest_path = cache_dir / "manifest.json"
    if raw_path.exists() and not refresh:
        logger.info("cache hit: %s", raw_path)
        return raw_path
    response = client.session.get(url, timeout=client.timeout_s)
    response.raise_for_status()
    raw_path.write_bytes(response.content)
    manifest_path.write_text(
        json.dumps({"url": url, "bytes": len(response.content)}, indent=2) + "\n"
    )
    logger.info("fetched %s (%d bytes)", url, len(response.content))
    return raw_path


def embed_school_calendar(
    raw: pd.DataFrame, out_csv: Path = SCHOOL_VACATIONS_CSV
) -> pd.DataFrame:
    """Validate (strict pydantic) and embed the raw calendar into the package."""
    clean = validate_school_vacations(raw)
    clean = clean.loc[clean.index >= EMBED_START]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.reset_index().to_csv(out_csv, index=False)
    logger.info(
        "embedded %d days (%s -> %s) into %s",
        len(clean),
        clean.index.min().date(),
        clean.index.max().date(),
        out_csv,
    )
    return clean
