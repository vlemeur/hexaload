"""Serve the feature table to walk-forward folds, masked at the fold origin.

``fold_features`` applies the as-of mask (target-reading columns become NaN
beyond their availability) and asserts the anti-leakage contract: exogenous
columns must cover the whole test block.
"""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from hexaload.features import apply_as_of

from .walkforward import Split

logger = logging.getLogger(__name__)


def fold_features(X: pd.DataFrame, manifest: dict, split: Split) -> pd.DataFrame:
    """Feature table for one fold: masked at as_of = end of the train block."""
    as_of = cast(pd.Timestamp, split.train.max())
    X_fold = apply_as_of(X, manifest, as_of)
    exo = [c for c, m in manifest.items() if not m.get("reads_target")]
    X_test = X_fold.loc[split.test]
    n_nan_exo = int(X_test[exo].isna().sum().sum())
    if n_nan_exo:
        raise AssertionError(
            f"fold {split.test_start.date()}: {n_nan_exo} NaN in exogenous "
            "features inside the test block — they must cover the horizon"
        )
    load_cols = [c for c in X_test.columns if c not in exo]
    n_nan_load = int(X_test[load_cols].isna().sum().sum())
    if n_nan_load:
        # routine: the target has a handful of gaps (22 NaN / 129k rows) and the
        # stitch boundary; the model imputes them — not worth a WARNING per fold
        logger.debug(
            "fold %s: %d NaN in load features (target gaps) — imputed by the model",
            split.test_start.date(),
            n_nan_load,
        )
    return X_fold
