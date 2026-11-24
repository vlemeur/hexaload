"""Feature-table assembly + end-to-end truncation no-leakage test (offline)."""

from pathlib import Path

import pandas as pd
import pytest

from hexaload.features import (
    CAL_COLUMNS,
    CYC_COLUMNS,
    assert_horizon_safe,
    build_feature_table,
)


def hourly(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def test_build_feature_table_shape_and_manifest(school_csv: Path) -> None:
    idx = hourly(24 * 400)
    X, manifest = build_feature_table(idx, school_csv=school_csv)
    assert X.index.equals(idx)
    assert set(X.columns) == set(CAL_COLUMNS) | set(CYC_COLUMNS)
    assert all(m["delay_h"] == 0 for m in manifest.values())
    assert {m["block"] for m in manifest.values()} == {"calendar", "cyclic"}
    assert_horizon_safe(manifest, horizon=24)


def test_as_of_masks_deterministic_features(school_csv: Path) -> None:
    idx = hourly(24 * 60)
    X_full, _ = build_feature_table(idx, school_csv=school_csv)
    as_of = idx[24 * 30]
    X, _ = build_feature_table(idx, as_of=as_of, school_csv=school_csv)
    assert X.loc[:as_of].notna().all().all()
    assert X.loc[as_of + pd.Timedelta(hours=1) :].isna().all().all()
    pd.testing.assert_frame_equal(X.loc[:as_of], X_full.loc[:as_of], check_dtype=False)


def test_truncation_no_leakage_end_to_end(school_csv: Path) -> None:
    """build(idx[:T], as_of=T) must equal build(idx_full) on t <= T."""
    idx = hourly(24 * 400)
    X_full, _ = build_feature_table(idx, school_csv=school_csv)
    T = idx[24 * 200]
    X_part, _ = build_feature_table(idx[: 24 * 200 + 1], as_of=T, school_csv=school_csv)
    pd.testing.assert_frame_equal(X_part, X_full.loc[:T], check_dtype=False)


def test_unknown_block_raises(school_csv: Path) -> None:
    with pytest.raises(ValueError, match="unknown feature block"):
        build_feature_table(hourly(48), blocks=("nope",), school_csv=school_csv)


def test_manifest_detects_unsafe_load_delay() -> None:
    manifest = {"lag_12": {"block": "lags", "delay_h": 12}}
    with pytest.raises(ValueError, match="leakage risk"):
        assert_horizon_safe(manifest, horizon=24)
    assert_horizon_safe({"lag_24": {"block": "lags", "delay_h": 24}}, horizon=24)
    # weather/calendar blocks are never load-derived: delay 0 is fine
    assert_horizon_safe({"wx_t": {"block": "weather", "delay_h": 0}}, horizon=24)
