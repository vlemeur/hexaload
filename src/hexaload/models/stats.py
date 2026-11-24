"""Statistical forecasters (ETS / Theta) via sktime.

Univariate for now (no xreg): these models only consume ``y_train``. The
underlying estimators are the statsmodels-backed sktime forecasters (hard
dependency of the project, see pyproject.toml).

AutoARIMA was dropped: the stepwise search on a year of hourly data costs
minutes per fit, which is not tractable over dozens of walk-forward folds.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.theta import ThetaForecaster

from .base import DEFAULT_QUANTILES, BaseForecaster


def _hourly_grid(y: pd.Series) -> pd.Series:
    """Reindex on a complete hourly grid (``freq="h"``); holes interpolated.

    sktime/statsmodels require a frequency on the index (the stitched eco2mix
    index has ``freq=None`` and may contain holes) and warn when converting
    tz-aware indexes to periods — so the grid is naive UTC (hexaload stores
    UTC everywhere; the tz is restored on the forecast output).
    """
    if y.index.tz is not None:
        y = y.tz_localize(None)
    idx = pd.date_range(y.index.min(), y.index.max(), freq="h")
    out = y.reindex(idx).interpolate(method="time")
    return out.dropna()


class SktimeForecaster(BaseForecaster):
    """Adapt a univariate sktime forecaster to the hexaload interface."""

    def __init__(self, forecaster: Any) -> None:
        self._forecaster = forecaster
        self._fitted: Any = None

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> SktimeForecaster:
        """Fit the wrapped forecaster on ``y_train`` (exogenous X not consumed yet)."""
        self._fitted = self._forecaster.fit(_hourly_grid(y_train))
        return self

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        """Point forecast over ``test`` via an absolute ForecastingHorizon.

        The fh is built on naive UTC (sktime drops tz when converting to
        periods); the returned Series is re-indexed on the tz-aware ``test``.
        """
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        test = pd.DatetimeIndex(test)
        fh = ForecastingHorizon(test.tz_localize(None), is_relative=False)
        pred = self._fitted.predict(fh=fh)
        return pd.Series(pred.to_numpy(), index=test, name="y_pred")

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Quantiles from the wrapped sktime forecaster (analytic intervals for
        statsmodels-backed models), re-indexed on the tz-aware ``test``."""
        if self._fitted is None:
            raise RuntimeError("call fit() before predict_quantiles()")
        test = pd.DatetimeIndex(test)
        fh = ForecastingHorizon(test.tz_localize(None), is_relative=False)
        q = self._fitted.predict_quantiles(fh=fh, alpha=list(quantiles))
        # sktime returns a DataFrame (time x levels) with MultiIndex columns
        # (variable, alpha) on a naive index — normalize to q<level> columns
        var = q.columns.get_level_values(0)[0]
        vals = q.loc[:, [(var, a) for a in quantiles]].to_numpy(dtype=float)
        return pd.DataFrame(
            {f"q{a:g}": vals[:, k] for k, a in enumerate(quantiles)},
            index=test,
        )


class SktimeAutoETS(SktimeForecaster):
    """ETS with fixed additive components (sktime, statsmodels-backed).

    ``auto=False`` on purpose: the automatic grid search over error/trend/
    seasonal combinations does not converge reliably on long hourly series
    and is very slow; ETS(A,A,A) with the daily period is the robust,
    deterministic choice for national load.
    """

    def __init__(self, *, sp: int = 24) -> None:
        super().__init__(
            AutoETS(auto=False, error="add", trend="add", seasonal="add", sp=sp)
        )


class SktimeTheta(SktimeForecaster):
    """Theta forecaster (sktime, statsmodels-backed)."""

    def __init__(self, *, sp: int = 24) -> None:
        super().__init__(ThetaForecaster(sp=sp))
