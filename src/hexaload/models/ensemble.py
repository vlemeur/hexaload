"""Online convex ensemble of feature models (ML-Poly-style weights).

The ensemble is just another backtest row: it is fitted inside each
walk-forward train fold only. Member models are fit on the full train fold
(they keep their own internal validation tail for early stopping /
conformal); the convex weights are learned online (Hedge / exponential
weights) on the LAST ``val_hours`` of the train fold, with a small 1-D grid
over the learning rate ``eta`` picked by validation MAE. No test-fold
information is ever touched.

Point forecast = sum_i w_i * pred_i. Quantiles = split-conformal around the
ensemble point forecast, calibrated on the ensemble residuals of that same
train-fold tail (same scheme as GBMBase).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from hexaload.metrics import quantile_frame

from .base import DEFAULT_QUANTILES, BaseForecaster

DEFAULT_ETAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


class EnsembleFeatures(BaseForecaster):
    """Convex combination of feature models with weights learned on the
    train-fold tail."""

    def __init__(
        self,
        members: Iterable[Any],
        *,
        val_hours: int = 24 * 14,
        etas: tuple[float, ...] = DEFAULT_ETAS,
    ) -> None:
        self._members = list(members)
        if not self._members:
            raise ValueError("EnsembleFeatures needs at least one member")
        self._val_hours = val_hours
        self._etas = tuple(etas)
        self._weights: np.ndarray | None = None
        self._resid: np.ndarray | None = None
        self._fitted: list[Any] = []

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> EnsembleFeatures:
        if X_train is None:
            raise ValueError("EnsembleFeatures is feature-based: X_train is required")
        y = y_train.reindex(X_train.index)
        mask = y.notna()
        X, y = X_train.loc[mask], y.loc[mask]
        if len(X) < self._val_hours + 24:
            raise ValueError(
                f"train fold too short for a {self._val_hours}h tail ({len(X)} rows)"
            )
        X_tail, y_tail = X.iloc[-self._val_hours :], y.iloc[-self._val_hours :]

        fitted = [m.fit(y_train, X_train) for m in self._members]
        self._fitted = fitted
        # member predictions on the weight-learning tail
        P = np.column_stack(
            [np.asarray(m.predict(y_tail.index, X_tail), dtype=float) for m in fitted]
        )
        r = y_tail.to_numpy(dtype=float)

        n = len(fitted)
        best_w, best_mae = None, np.inf
        for eta in self._etas:
            # Hedge in log-space with max-shift: losses are in MW, so
            # exp(-eta*loss) would underflow to 0 without the shift
            lw = np.full(n, -np.log(n))
            cum = 0.0
            for t in range(len(r)):
                w = np.exp(lw - lw.max())
                w = w / w.sum()
                ens = float(w @ P[t])
                cum += abs(r[t] - ens)
                lw = lw - eta * np.abs(r[t] - P[t])
            w = np.exp(lw - lw.max())
            w = w / w.sum()
            mae = cum / len(r)
            if mae < best_mae:
                best_mae, best_w = mae, w
        self._weights = best_w
        # split-conformal residuals of the ENSEMBLE on the same tail
        ens_tail = P @ self._weights
        self._resid = r - ens_tail
        return self

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        if X_future is None or self._weights is None:
            raise RuntimeError("call fit() with X first")
        test = pd.DatetimeIndex(test)
        Xt = X_future.reindex(test)
        pred = sum(
            w * np.asarray(m.predict(test, Xt), dtype=float)
            for w, m in zip(self._weights, self._fitted, strict=False)
        )
        return pd.Series(np.clip(pred, 0.0, None), index=test, name="y_pred")

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        if self._resid is None:
            raise RuntimeError("call fit() first")
        return quantile_frame(self.predict(test, X_future), self._resid, quantiles)
