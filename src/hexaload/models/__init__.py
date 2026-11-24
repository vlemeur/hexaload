"""Model families — one common interface.

``.fit(y_train, X_train=None)`` / ``.predict(test, X_future=None)``.
"""

from ._device import default_device, resolve_device
from .base import DEFAULT_QUANTILES, BaseForecaster
from .gbm import GBMBase, HGBFeatures, XGBFeatures
from .linear import RidgeFeatures
from .stats import SktimeAutoETS, SktimeTheta

__all__ = [
    "DEFAULT_QUANTILES",
    "BaseForecaster",
    "GBMBase",
    "HGBFeatures",
    "RidgeFeatures",
    "SktimeAutoETS",
    "SktimeTheta",
    "XGBFeatures",
    "default_device",
    "resolve_device",
]
