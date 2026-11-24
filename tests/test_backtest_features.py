"""fold_features: per-fold as-of masking of the feature table (offline)."""

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from hexaload.backtest.features import fold_features
from hexaload.backtest.walkforward import walk_forward_splits
from hexaload.features import build_feature_table


def test_fold_features_masks_lags_and_covers_test_block(
    hourly_idx: pd.DatetimeIndex,
    school_csv: Path,
    wx_national: pd.DataFrame,
    y_load: pd.Series,
) -> None:
    X, manifest = build_feature_table(
        hourly_idx, school_csv=school_csv, wx=wx_national, y=y_load
    )
    splits = walk_forward_splits(
        y_load,
        start=str(hourly_idx[0].date()),
        end=str(hourly_idx[-1].date()),
        horizon=24,
        window_length=24 * 20,
        step_length=24 * 5,
    )
    split = splits[0]
    X_fold = fold_features(X, manifest, split)
    as_of = cast(pd.Timestamp, split.train.max())

    # exogenous columns cover the whole test block
    exo = [c for c, m in manifest.items() if not m["reads_target"]]
    assert not X_fold.loc[split.test, exo].isna().any().any()
    # lag_load_24h available on the whole 24h test block (delay 24 >= horizon)
    assert X_fold.loc[split.test, "lag_load_24h"].notna().all()
    # ...and masked right after as_of + 24h
    after = X_fold.loc[as_of + pd.Timedelta(hours=25) :, "lag_load_24h"]
    assert after.isna().all()
    # no target value after as_of leaks into the available load features:
    # compare against a table built from a future-perturbed target
    y_future = y_load.copy()
    y_future.loc[as_of + pd.Timedelta(hours=1) :] = 1e9
    X2, manifest2 = build_feature_table(
        hourly_idx, school_csv=school_csv, wx=wx_national, y=y_future
    )
    X_fold2 = fold_features(X2, manifest2, split)
    load_cols = [c for c, m in manifest.items() if m["reads_target"]]
    pd.testing.assert_frame_equal(
        X_fold.loc[:, load_cols], X_fold2.loc[:, load_cols], check_dtype=False
    )


def test_fold_features_raises_when_exogenous_gap(
    hourly_idx: pd.DatetimeIndex,
    school_csv: Path,
    wx_national: pd.DataFrame,
    y_load: pd.Series,
) -> None:
    X, manifest = build_feature_table(
        hourly_idx, school_csv=school_csv, wx=wx_national, y=y_load
    )
    splits = walk_forward_splits(
        y_load,
        start=str(hourly_idx[0].date()),
        end=str(hourly_idx[-1].date()),
        horizon=24,
        window_length=24 * 20,
        step_length=24 * 5,
    )
    # punch a hole in an exogenous column inside the first test block
    X_bad = X.copy()
    X_bad.loc[splits[0].test[0], "cal_ferie"] = float("nan")
    with pytest.raises(AssertionError, match="exogenous"):
        fold_features(X_bad, manifest, splits[0])
