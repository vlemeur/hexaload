"""Linear model family: ridge regression on the feature table (scikit-learn).

First model of the benchmark beyond the naive baselines — its purpose is to
prove the feature table is consumable end-to-end by a walk-forward fold
(features masked at the fold origin, no leakage). Deterministic, offline.
Predictions are clipped at 0 MW (load cannot be negative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RidgeFeatures:
    """sklearn pipeline (median-impute -> standardize -> Ridge) on features.

    Common hexaload model interface: ``.fit(X_train, y_train) -> self``,
    ``.predict(X_future) -> pd.Series`` (point forecast; quantiles come with
    the probabilistic families).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self._pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        self._cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeFeatures:
        """Fit on X/y aligned on X's index; NaN rows handled by the imputer."""
        y_aligned = y.reindex(X.index)
        mask = y_aligned.notna()
        if not mask.any():
            raise ValueError("no non-NaN target rows to fit on")
        self._cols = list(X.columns)
        self._pipe.fit(X.loc[mask, self._cols], y_aligned.loc[mask])
        return self

    def predict(self, X_future: pd.DataFrame) -> pd.Series:
        """Point forecast on ``X_future`` (already masked at the fold origin)."""
        if self._cols is None:
            raise RuntimeError("call fit() before predict()")
        missing = [c for c in self._cols if c not in X_future.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:5]}")
        pred = self._pipe.predict(X_future[self._cols])
        return pd.Series(np.clip(pred, 0.0, None), index=X_future.index, name="y_pred")
