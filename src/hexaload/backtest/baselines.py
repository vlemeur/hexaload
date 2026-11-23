"""Naive baselines and the RTE reference — pure pandas, leakage-safe.

Each predictor only reads values at timestamps <= the end of the training
fold (asserted where a lag is involved).
"""

from __future__ import annotations

import pandas as pd


def predict_naive(y_train: pd.Series, test: pd.DatetimeIndex) -> pd.Series:
    """Last observed value carried forward over the horizon."""
    return pd.Series(y_train.iloc[-1], index=test, name="y_pred")


def predict_seasonal_naive(
    y: pd.Series,
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    *,
    sp: int,
) -> pd.Series:
    """y(t) = y(t - sp), with the source timestamp asserted inside the train fold."""
    pos = y.index.get_indexer(test)
    src = pos - sp
    if (src < 0).any():
        raise ValueError(f"not enough history for seasonal-naive sp={sp}")
    src_ts = y.index[src]
    if (src_ts > train.max()).any():
        raise AssertionError("leakage: seasonal-naive source after train end")
    return pd.Series(y.to_numpy()[src], index=test, name="y_pred")


def predict_rte_prevision_j1(
    prevision_j1: pd.Series, test: pd.DatetimeIndex
) -> pd.Series:
    """RTE's own D-1 forecast (published ~24 h ahead): the absolute reference."""
    return prevision_j1.reindex(test).rename("y_pred")
