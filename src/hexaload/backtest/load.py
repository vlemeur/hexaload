"""Shared offline data loading for the scripts (backtest, tuning, forecast).

Everything reads the on-disk cache under ``data/processed`` — no network, so
the backtest and the tuning both run fully offline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from hexaload.data.fetch import stitch_eco2mix

logger = logging.getLogger(__name__)

DATA = Path("data/processed")
CONS_DEF_PARQUET = DATA / "eco2mix-national-cons-def-hourly.parquet"
TR_PARQUET = DATA / "eco2mix-national-tr-hourly.parquet"
WX_PARQUET = DATA / "openmeteo-icon_eu-hourly-national.parquet"
FEATURES_PARQUET = DATA / "features-latest.parquet"
FEATURES_MANIFEST = DATA / "features-latest.manifest.json"


def load_features() -> tuple[pd.DataFrame | None, dict | None]:
    """Materialized feature table + manifest (scripts/build_features.py)."""
    if not FEATURES_PARQUET.exists():
        logger.warning(
            "no feature table (%s missing); run scripts/build_features.py — "
            "feature-based models are skipped",
            FEATURES_PARQUET,
        )
        return None, None
    X = pd.read_parquet(FEATURES_PARQUET)
    manifest = json.loads(FEATURES_MANIFEST.read_text())["features"]
    logger.info(
        "features: %d cols x %d rows (%d target-reading)",
        X.shape[1],
        len(X),
        sum(1 for m in manifest.values() if m.get("reads_target")),
    )
    return X, manifest


def load_targets() -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Stitched y/ref (cons-def + real-time tail), weather, and the source mask."""
    cons_def = pd.read_parquet(CONS_DEF_PARQUET)
    tr = pd.read_parquet(TR_PARQUET) if TR_PARQUET.exists() else pd.DataFrame()
    if tr.empty:
        logger.warning(
            "no real-time tail (%s missing); using cons-def only", TR_PARQUET
        )
        eco = cons_def.copy()
        eco["source"] = "cons-def"
    else:
        eco = stitch_eco2mix(cons_def, tr)
        boundary = eco.index[eco["source"] == "tr"][0]
        logger.info(
            "stitched eco2mix: cons-def -> %s, tr from %s", eco.index.max(), boundary
        )
    y = eco["consommation"]
    ref = eco["prevision_j1"]
    source = eco["source"]
    # Exogenous archive forecasts (leakage-safe by construction); not used by
    # the naive baselines yet, loaded to assert the pipeline is complete.
    wx = pd.read_parquet(WX_PARQUET)
    return y, ref, wx, source


def exogenous_columns(manifest: dict | None) -> list[str]:
    """Feature columns that do not read the target (safe as known-future)."""
    if manifest is None:
        return []
    return [c for c, m in manifest.items() if not m.get("reads_target")]
