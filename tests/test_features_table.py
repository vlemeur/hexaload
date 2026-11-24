"""Feature-table assembly + end-to-end no-leakage tests (offline)."""

from pathlib import Path

import pandas as pd
import pytest

from hexaload.features import (
    CAL_COLUMNS,
    CYC_COLUMNS,
    assert_horizon_safe,
    build_feature_table,
)


def test_build_full_table_shape_and_manifest(
    hourly_idx: pd.DatetimeIndex,
    school_csv: Path,
    wx_national: pd.DataFrame,
    y_load: pd.Series,
) -> None:
    X, manifest = build_feature_table(
        hourly_idx, school_csv=school_csv, wx=wx_national, y=y_load
    )
    assert X.index.equals(hourly_idx)
    assert set(CAL_COLUMNS) | set(CYC_COLUMNS) <= set(X.columns)
    assert any(c.startswith("wx_") for c in X.columns)
    assert any(c.startswith("lag_") for c in X.columns)
    assert all(
        m["delay_h"] == 0 and not m["reads_target"]
        for m in manifest.values()
        if m["block"] != "lags"
    )
    assert all(
        m["delay_h"] >= 24 and m["reads_target"]
        for m in manifest.values()
        if m["block"] == "lags"
    )
    assert_horizon_safe(manifest, horizon=24)


def test_missing_inputs_raise(
    hourly_idx: pd.DatetimeIndex, school_csv: Path, wx_national: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="requires wx"):
        build_feature_table(hourly_idx, school_csv=school_csv)
    with pytest.raises(ValueError, match="requires y"):
        build_feature_table(hourly_idx, school_csv=school_csv, wx=wx_national)


def test_as_of_masks_by_delay(
    hourly_idx: pd.DatetimeIndex,
    school_csv: Path,
    wx_national: pd.DataFrame,
    y_load: pd.Series,
) -> None:
    X_full, manifest = build_feature_table(
        hourly_idx, school_csv=school_csv, wx=wx_national, y=y_load
    )
    as_of = hourly_idx[24 * 20]
    X, _ = build_feature_table(
        hourly_idx, as_of=as_of, school_csv=school_csv, wx=wx_national, y=y_load
    )
    # exogenous columns (reads_target=False): known in advance, available on
    # the WHOLE horizon including the test block (past the 24h lag warm-up)
    det = [c for c, m in manifest.items() if not m["reads_target"]]
    warm = hourly_idx[24]
    assert X.loc[warm:, det].notna().all().all()
    # target-reading columns are masked beyond their availability
    load = [c for c, m in manifest.items() if m["reads_target"]]
    assert X["lag_load_24h"].loc[as_of + pd.Timedelta(hours=25) :,].isna().all()
    assert load
    # lag_load_24h (delay 24): known up to as_of + 24h (after its 24h warm-up)
    warm = hourly_idx[24]
    assert X["lag_load_24h"].loc[warm : as_of + pd.Timedelta(hours=24)].notna().all()
    assert X["lag_load_24h"].loc[as_of + pd.Timedelta(hours=25) :].isna().all()
    pd.testing.assert_frame_equal(
        X.loc[:as_of, det], X_full.loc[:as_of, det], check_dtype=False
    )


def test_future_perturbation_no_leak_end_to_end(
    hourly_idx: pd.DatetimeIndex,
    school_csv: Path,
    wx_national: pd.DataFrame,
    y_load: pd.Series,
) -> None:
    """Overwriting the target after as_of changes nothing in the masked table."""
    as_of = hourly_idx[24 * 30]
    y_future = y_load.copy()
    y_future.loc[as_of + pd.Timedelta(hours=1) :] = 1e9
    base, _ = build_feature_table(
        hourly_idx, as_of=as_of, school_csv=school_csv, wx=wx_national, y=y_load
    )
    pert, _ = build_feature_table(
        hourly_idx, as_of=as_of, school_csv=school_csv, wx=wx_national, y=y_future
    )
    pd.testing.assert_frame_equal(base, pert, check_dtype=False)


def test_truncation_no_leakage_end_to_end(
    hourly_idx: pd.DatetimeIndex,
    school_csv: Path,
    wx_national: pd.DataFrame,
    y_load: pd.Series,
) -> None:
    """build(idx[:T], as_of=T) equals build(idx_full) on t <= T for the
    pointwise delay-0 columns (calendar/cyclic). Windowed aggregates (day
    min/max, rolling) legitimately differ at the truncated border — in the
    real backtest the table is built once over the full period and masked
    with as_of, never truncated."""
    X_full, _ = build_feature_table(
        hourly_idx, school_csv=school_csv, wx=wx_national, y=y_load
    )
    T = hourly_idx[24 * 20]
    k = hourly_idx.get_loc(T)
    X_part, _ = build_feature_table(
        hourly_idx[: k + 1],
        as_of=T,
        school_csv=school_csv,
        wx=wx_national.loc[:T],
        y=y_load.loc[:T],
    )
    pointwise = [c for c in X_full.columns if c.startswith(("cal_", "cyc_"))]
    pd.testing.assert_frame_equal(
        X_part[pointwise], X_full.loc[:T, pointwise], check_dtype=False
    )


def test_unknown_block_raises(hourly_idx: pd.DatetimeIndex, school_csv: Path) -> None:
    with pytest.raises(ValueError, match="unknown feature block"):
        build_feature_table(hourly_idx, blocks=("nope",), school_csv=school_csv)
