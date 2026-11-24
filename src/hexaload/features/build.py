"""Build the materialized feature table from the on-disk cache (offline).

``scripts/build_features.py`` is the thin CLI over :func:`build`. Reads the
eco2mix cache (cons-def + real-time tail, stitched) and the Open-Meteo grid
cache, aggregates weather to national hourly (population-weighted when
``features/assets/grid_weights.csv`` exists, plain grid mean otherwise), builds
every feature block and writes ``features-<digest>.parquet`` plus its
manifest (delays per column, sha256 of every input, horizon). Stable aliases
``features-latest.parquet`` / ``features-latest.manifest.json`` are refreshed
for the backtest. All under ``data/processed/``: generated, gitignored,
never hand-edited.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..data.fetch import CONS_DEF_DATASET, TR_DATASET, read_cache, stitch_eco2mix
from ..data.validate import NATURE_TR, aggregate_hourly, validate_raw
from .calendar import SCHOOL_VACATIONS_CSV
from .table import DEFAULT_BLOCKS, DEFAULT_HORIZON, build_feature_table
from .weather import GRID_WEIGHTS_CSV, load_grid_weights, national_weather_frame

logger = logging.getLogger(__name__)

OPENMETEO_DATASET = "openmeteo-icon_eu"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_targets(cache_dir: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Stitched hourly eco2mix (y, prevision_j1, source mask) from the cache."""
    cons_def, _ = validate_raw(read_cache(CONS_DEF_DATASET, cache_dir))
    eco = aggregate_hourly(cons_def)
    eco["source"] = "cons-def"
    tr_manifest = cache_dir / TR_DATASET / "manifest.json"
    if tr_manifest.exists():
        tr, _ = validate_raw(read_cache(TR_DATASET, cache_dir))
        eco = stitch_eco2mix(eco, aggregate_hourly(tr, natures=(NATURE_TR,)))
    return eco["consommation"], eco["prevision_j1"], eco[["source"]]


def load_weather(cache_dir: Path) -> pd.DataFrame:
    """National hourly weather frame, population-weighted when weights exist."""
    raw = read_cache(OPENMETEO_DATASET, cache_dir)
    return national_weather_frame(raw, load_grid_weights())


def build(
    cache_dir: Path, out_dir: Path, *, horizon: int = DEFAULT_HORIZON
) -> tuple[Path, Path]:
    """Build the feature table and write parquet + manifest; returns paths."""
    y, prevision_j1, source = load_targets(cache_dir)
    wx = load_weather(cache_dir)
    idx = y.index
    X, manifest = build_feature_table(
        idx,
        wx=wx,
        y=y,
        prevision_j1=prevision_j1,
        horizon=horizon,
        blocks=DEFAULT_BLOCKS,
    )
    inputs = {
        "eco2mix_manifest": _file_sha256(
            cache_dir / CONS_DEF_DATASET / "manifest.json"
        ),
        "tr_manifest": _file_sha256(cache_dir / TR_DATASET / "manifest.json")
        if (cache_dir / TR_DATASET / "manifest.json").exists()
        else None,
        "openmeteo_manifest": _file_sha256(
            cache_dir / OPENMETEO_DATASET / "manifest.json"
        ),
        "grid_weights": _file_sha256(GRID_WEIGHTS_CSV)
        if GRID_WEIGHTS_CSV.exists()
        else None,
        "school_vacations": _file_sha256(SCHOOL_VACATIONS_CSV),
    }
    digest = hashlib.sha256(
        json.dumps(
            {"inputs": inputs, "rows": len(X), "cols": list(X.columns)},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"features-{digest}.parquet"
    out_manifest = out_dir / f"features-{digest}.manifest.json"
    X.to_parquet(out_parquet, index=True)
    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "rows": len(X),
        "horizon_h": horizon,
        "blocks": list(DEFAULT_BLOCKS),
        "inputs_sha256": inputs,
        "features": manifest,
        "source_tail": source["source"].iloc[-1],
        "range": [str(idx.min()), str(idx.max())],
    }
    out_manifest.write_text(json.dumps(meta, indent=2) + "\n")
    # stable aliases for the backtest to pick up
    (out_dir / "features-latest.parquet").write_bytes(out_parquet.read_bytes())
    (out_dir / "features-latest.manifest.json").write_text(out_manifest.read_text())
    logger.info(
        "wrote %s (%d rows x %d cols) + manifest; digest %s",
        out_parquet,
        len(X),
        X.shape[1],
        digest,
    )
    return out_parquet, out_manifest
