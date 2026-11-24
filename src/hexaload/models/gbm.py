"""Gradient-boosted trees on the feature table (sklearn-API GBMs).

Shared skeleton (:class:`GBMBase`): early stopping on an explicit train-tail
validation split (handed to the underlying library so nothing is shuffled —
the test block is never touched) + split-conformal quantiles from that same
tail. Concrete families: :class:`HGBFeatures` (sklearn) and ``XGBFeatures``
(``xgb.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from hexaload.metrics import quantile_frame

from .base import DEFAULT_QUANTILES, BaseForecaster


class GBMBase(BaseForecaster):
    """Shared fit/predict/predict_quantiles for sklearn-API GBMs on features."""

    def __init__(self, *, val_hours: int = 24 * 14, random_state: int = 0) -> None:
        self._val_hours = val_hours
        self._random_state = random_state
        self._model: Any = None
        self._cols: list[str] | None = None
        self._resid: np.ndarray | None = None

    def _new_model(self) -> Any:
        """Instantiate the underlying regressor (library-specific)."""
        raise NotImplementedError

    def _fit_model(
        self,
        model: Any,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Fit with early stopping on the validation tail (library-specific)."""
        raise NotImplementedError

    def _predict_raw(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict(X), dtype=float)

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> GBMBase:
        """Fit with early stopping on the train-fold tail (validation split)."""
        if X_train is None:
            raise ValueError(
                f"{type(self).__name__} is feature-based: X_train is required"
            )
        y = y_train.reindex(X_train.index)
        mask = y.notna()
        X, y = X_train.loc[mask], y.loc[mask]
        if len(X) < self._val_hours + 24:
            raise ValueError(
                f"train fold too short for a {self._val_hours}h validation tail "
                f"({len(X)} rows)"
            )
        X_fit, X_val = X.iloc[: -self._val_hours], X.iloc[-self._val_hours :]
        y_fit, y_val = y.iloc[: -self._val_hours], y.iloc[-self._val_hours :]
        model = self._new_model()
        self._fit_model(model, X_fit, y_fit, X_val, y_val)
        self._model = model
        self._cols = list(X_train.columns)
        # split-conformal calibration residuals on the validation tail
        self._resid = (y_val - self._predict_raw(model, X_val)).to_numpy(dtype=float)
        return self

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        """Point forecast over ``test``; NaN features are handled natively."""
        model, cols = self._model, self._cols
        if model is None or cols is None:
            raise RuntimeError("call fit() before predict()")
        if X_future is None:
            raise ValueError(
                f"{type(self).__name__} is feature-based: X_future is required"
            )
        test = pd.DatetimeIndex(test)
        Xt = X_future.reindex(test)
        missing = [c for c in cols if c not in Xt.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:5]}")
        pred = self._predict_raw(model, Xt[cols])
        return pd.Series(np.clip(pred, 0.0, None), index=test, name="y_pred")

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Split-conformal quantiles: point forecast + quantiles of the
        validation-tail residuals (same split as early stopping, no leakage)."""
        pred = self.predict(test, X_future)
        if self._resid is None:
            raise RuntimeError("call fit() before predict_quantiles()")
        return quantile_frame(pred, self._resid, quantiles)


class HGBFeatures(GBMBase):
    """HistGradientBoostingRegressor on the feature table (pure sklearn)."""

    def __init__(
        self,
        *,
        max_iter: int = 500,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        val_hours: int = 24 * 14,
        random_state: int = 0,
    ) -> None:
        super().__init__(val_hours=val_hours, random_state=random_state)
        self._params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
        }

    def _new_model(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            early_stopping=True,
            validation_fraction=None,  # explicit X_val/y_val: no shuffle
            random_state=self._random_state,
            **self._params,
        )

    def _fit_model(
        self,
        model: HistGradientBoostingRegressor,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
