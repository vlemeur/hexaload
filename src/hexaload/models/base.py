"""Common model interface for the hexaload benchmark.

Every model — baseline, feature-based, statistical, TSFM — implements:

    .fit(y_train, X_train=None) -> self
    .predict(test, X_future=None) -> pd.Series

- ``y_train``: target series on the train fold (UTC DatetimeIndex);
- ``X_train``: feature table restricted to the train fold, already masked at
  the fold origin by ``fold_features`` (feature-based models only);
- ``test``: DatetimeIndex of the horizon block to predict;
- ``X_future``: feature table restricted to the test block (feature-based
  models only; exogenous columns must cover the whole block).

Point forecasts only for now; probabilistic families will add
``predict_quantiles``/``sample`` and be scored with CRPS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# default quantile grid for predict_quantiles / CRPS sampling: 0.05 .. 0.95
DEFAULT_QUANTILES: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))


class BaseForecaster(ABC):
    """Common interface: fit on a train fold, predict a test block."""

    @abstractmethod
    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> BaseForecaster:
        """Fit on the train fold. Feature-based models require ``X_train``."""

    @abstractmethod
    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        """Point forecast over ``test`` (coerced to a UTC DatetimeIndex).

        Returns a Series named ``y_pred`` indexed by ``test``. Feature-based
        models require ``X_future`` covering ``test``.
        """

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Quantile forecast over ``test``: DataFrame indexed by ``test``, one
        column per level (``q0.05``, ``q0.1``, ...).

        Point-only models raise NotImplementedError and are scored without CRPS.
        """
        raise NotImplementedError
