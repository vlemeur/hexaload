"""Model families — one common interface.

``.fit(y_train, X_train=None)`` / ``.predict(test, X_future=None)``.
"""

from .base import DEFAULT_QUANTILES, BaseForecaster
from .gbm import HGBFeatures
from .linear import RidgeFeatures
from .stats import SktimeAutoETS, SktimeTheta

__all__ = [
    "DEFAULT_QUANTILES",
    "BaseForecaster",
    "HGBFeatures",
    "RidgeFeatures",
    "SktimeAutoETS",
    "SktimeTheta",
]
