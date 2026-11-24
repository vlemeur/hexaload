"""Naive baselines and the RTE reference — pure pandas, leakage-safe.

Each predictor only reads values at timestamps <= the end of the training
fold (asserted where a lag is involved).
"""

from __future__ import annotations

import pandas as pd

from hexaload.metrics import quantile_frame


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


def _lag_residuals(y: pd.Series, sp: int) -> pd.Series:
    """``y(t) - y(t - sp hours)``, aligned by timestamp (robust to index holes)."""
    src = y.reindex(y.index - pd.Timedelta(hours=sp))
    src.index = y.index  # src[t] = y(t - sp h)
    return (y - src).dropna()


def naive_residuals(y_train: pd.Series) -> pd.Series:
    """One-step residuals of the naive forecast on the train fold (as-of)."""
    return _lag_residuals(y_train, 1)


def seasonal_residuals(y_train: pd.Series, *, sp: int) -> pd.Series:
    """Residuals of the seasonal-naive (period ``sp`` hours) on the train fold."""
    return _lag_residuals(y_train, sp)


def residual_quantiles(
    pred: pd.Series,
    residuals: pd.Series,
    quantiles: tuple[float, ...],
) -> pd.DataFrame:
    """Split-conformal quantile frame around a point forecast.

    The calibration residuals come from the TRAIN fold only — the test block
    is never used, so the intervals are leakage-free.
    """
    return quantile_frame(pred, residuals.dropna().to_numpy(dtype=float), quantiles)
