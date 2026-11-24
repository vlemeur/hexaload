"""XGBoost on the feature table (``xgboost`` sklearn API).

Same skeleton as :class:`~hexaload.models.gbm.HGBFeatures`
(:class:`~hexaload.models.gbm.GBMBase`): early stopping on the explicit
train-tail validation split (``eval_set``), NaN handled natively
(sparsity-aware), split-conformal quantiles from that tail. Predictions are
clipped at 0 MW.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .gbm import GBMBase


class XGBFeatures(GBMBase):
    """XGBRegressor on the feature table."""

    def __init__(
        self,
        *,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        val_hours: int = 24 * 14,
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(val_hours=val_hours, random_state=random_state)
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
        }
        self._es_rounds = early_stopping_rounds
        self._n_jobs = n_jobs

    def _new_model(self) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            early_stopping_rounds=self._es_rounds,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
            **self._params,
        )

    def _fit_model(
        self,
        model: XGBRegressor,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    def _predict_raw(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        # with early stopping, predict at the best iteration, not the last
        best = getattr(model, "best_iteration", None)
        if best is not None:
            return np.asarray(
                model.predict(X, iteration_range=(0, best + 1)), dtype=float
            )
        return np.asarray(model.predict(X), dtype=float)
