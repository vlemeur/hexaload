"""Gradient-boosted trees on the feature table (sklearn HistGradientBoostingRegressor).

Second feature-based family after the ridge. HGBR handles NaN features
natively (no imputer needed) and supports early stopping on an explicit
validation split. Anti-leakage: the validation split is the TAIL of the train
fold, passed as ``X_val``/``y_val`` so sklearn does not shuffle — the test
block is never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .base import BaseForecaster


class HGBFeatures(BaseForecaster):
    """HistGradientBoostingRegressor on the feature table.

    Common hexaload interface: ``.fit(y_train, X_train)`` and
    ``.predict(test, X_future)``.
    """

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
        self._params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
        }
        self._val_hours = val_hours
        self._random_state = random_state
        self._model: HistGradientBoostingRegressor | None = None
        self._cols: list[str] | None = None

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> HGBFeatures:
        """
        Fit on the train fold; the last ``val_hours`` rows are the validation tail.
        """
        if X_train is None:
            raise ValueError("HGBFeatures is feature-based: X_train is required")
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
        model = HistGradientBoostingRegressor(
            early_stopping=True,
            validation_fraction=None,  # explicit X_val/y_val below: no shuffle
            random_state=self._random_state,
            **self._params,
        )
        model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
        self._model = model
        self._cols = list(X_train.columns)
        return self

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        """Point forecast over ``test``; NaN features are handled natively by HGBR."""
        model, cols = self._model, self._cols
        if model is None or cols is None:
            raise RuntimeError("call fit() before predict()")
        if X_future is None:
            raise ValueError("HGBFeatures is feature-based: X_future is required")
        test = pd.DatetimeIndex(test)
        Xt = X_future.reindex(test)
        missing = [c for c in cols if c not in Xt.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:5]}")
        pred = model.predict(Xt[cols])
        return pd.Series(np.clip(pred, 0.0, None), index=test, name="y_pred")
