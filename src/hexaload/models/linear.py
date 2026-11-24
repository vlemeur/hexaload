"""Linear model family: ridge regression on the feature table (scikit-learn).

First model of the benchmark beyond the naive baselines — its purpose is to
prove the feature table is consumable end-to-end by a walk-forward fold
(features masked at the fold origin, no leakage). Deterministic, offline.
Predictions are clipped at 0 MW (load cannot be negative).

Probabilistic side: split conformal — the tail of the train fold is held out
as a calibration split, and ``predict_quantiles`` adds the empirical quantiles
of the calibration residuals to the point forecast (the test block is never
touched).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hexaload.metrics import quantile_frame

from .base import DEFAULT_QUANTILES, BaseForecaster


class RidgeFeatures(BaseForecaster):
    """sklearn pipeline (median-impute -> standardize -> Ridge) on features.

    Common hexaload interface: ``.fit(y_train, X_train)``,
    ``.predict(test, X_future)`` and ``.predict_quantiles(...)``.
    """

    def __init__(self, alpha: float = 1.0, val_hours: int = 24 * 14) -> None:
        self._pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        self._val_hours = val_hours
        self._cols: list[str] | None = None
        self._resid: np.ndarray | None = None

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> RidgeFeatures:
        """Fit on the train fold minus its calibration tail (last ``val_hours``)."""
        if X_train is None:
            raise ValueError("RidgeFeatures is feature-based: X_train is required")
        y_aligned = y_train.reindex(X_train.index)
        mask = y_aligned.notna()
        if not mask.any():
            raise ValueError("no non-NaN target rows to fit on")
        X, y = X_train.loc[mask], y_aligned.loc[mask]
        if len(X) < self._val_hours + 24:
            raise ValueError(
                f"train fold too short for a {self._val_hours}h calibration tail "
                f"({len(X)} rows)"
            )
        self._cols = list(X_train.columns)
        self._pipe.fit(X.iloc[: -self._val_hours], y.iloc[: -self._val_hours])
        val_pred = self._pipe.predict(X.iloc[-self._val_hours :])
        self._resid = (y.iloc[-self._val_hours :] - val_pred).to_numpy(dtype=float)
        return self

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        """Point forecast over ``test`` (features already masked at the fold origin)."""
        cols = self._cols
        if cols is None:
            raise RuntimeError("call fit() before predict()")
        if X_future is None:
            raise ValueError("RidgeFeatures is feature-based: X_future is required")
        test = pd.DatetimeIndex(test)
        Xt = X_future.reindex(test)
        missing = [c for c in cols if c not in Xt.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:5]}")
        pred = self._pipe.predict(Xt[cols])
        return pd.Series(np.clip(pred, 0.0, None), index=test, name="y_pred")

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Split-conformal quantiles: point forecast + quantiles of the
        calibration residuals (tail of the train fold)."""
        pred = self.predict(test, X_future)
        if self._resid is None:
            raise RuntimeError("call fit() before predict_quantiles()")
        return quantile_frame(pred, self._resid, quantiles)
