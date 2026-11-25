"""Post-hoc wrappers around any feature model:

- :class:`MondrianConformal` — split-conformal quantiles bucketed by LOCAL
  hour (Europe/Paris): tight intervals off-peak, wide on the evening peak.
  Falls back to global conformal quantiles for unseen hours.
- :class:`BiasEWMA` — subtracts an EWMA of the validation-tail residuals,
  removing systematic bias (e.g. GBMs under-shooting peaks).

Both only use the train fold (inner model + its validation tail); the test
block is never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DEFAULT_QUANTILES, BaseForecaster

PARIS = "Europe/Paris"


def _paris_hour(idx: pd.DatetimeIndex) -> np.ndarray:
    return pd.DatetimeIndex(idx).tz_convert(PARIS).hour.to_numpy()


class MondrianConformal(BaseForecaster):
    """Per-local-hour split-conformal intervals around an inner model."""

    def __init__(
        self, inner: BaseForecaster, *, val_hours: int = 24 * 14, min_bucket: int = 7
    ) -> None:
        self._inner = inner
        self._val_hours = val_hours
        self._min_bucket = min_bucket
        self._resid_by_hour: dict[int, np.ndarray] = {}
        self._resid_global: np.ndarray | None = None

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> MondrianConformal:
        self._inner.fit(y_train, X_train)
        y = y_train.reindex(X_train.index)
        mask = y.notna()
        X, yv = X_train.loc[mask], y.loc[mask]
        X_tail, y_tail = X.iloc[-self._val_hours :], yv.iloc[-self._val_hours :]
        resid = (y_tail - self._inner.predict(y_tail.index, X_tail)).to_numpy(
            dtype=float
        )
        hours = _paris_hour(y_tail.index)
        self._resid_global = resid
        for h in range(24):
            r = resid[hours == h]
            if len(r) >= self._min_bucket:  # enough samples for a per-hour bucket
                self._resid_by_hour[h] = r
        return self

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        return self._inner.predict(test, X_future)

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        pred = self.predict(test, X_future)
        test = pd.DatetimeIndex(test)
        hours = _paris_hour(test)
        cols = {}
        for a in quantiles:
            off = np.empty(len(test))
            for i, h in enumerate(hours):
                r = self._resid_by_hour.get(int(h), self._resid_global)
                off[i] = np.quantile(r, a, method="linear")
            cols[f"q{a:g}"] = pred.to_numpy() + off
        return pd.DataFrame(cols, index=test)


class BiasEWMA(BaseForecaster):
    """Subtract an EWMA of the validation-tail residuals (bias correction)."""

    def __init__(
        self, inner: BaseForecaster, *, val_hours: int = 24 * 14, alpha: float = 0.1
    ) -> None:
        self._inner = inner
        self._val_hours = val_hours
        self._alpha = alpha
        self._corr = 0.0

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> BiasEWMA:
        self._inner.fit(y_train, X_train)
        y = y_train.reindex(X_train.index)
        mask = y.notna()
        X, yv = X_train.loc[mask], y.loc[mask]
        X_tail, y_tail = X.iloc[-self._val_hours :], yv.iloc[-self._val_hours :]
        resid = (y_tail - self._inner.predict(y_tail.index, X_tail)).to_numpy(
            dtype=float
        )
        c = resid[0]
        for r in resid[1:]:
            c = self._alpha * r + (1 - self._alpha) * c
        self._corr = float(c)
        return self

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        # resid = y - pred, so the correction is ADDED to the prediction
        return self._inner.predict(test, X_future) + self._corr

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        q = self._inner.predict_quantiles(test, X_future, quantiles)
        return q + self._corr
