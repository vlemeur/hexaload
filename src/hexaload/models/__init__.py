"""Model families — one common interface.

``.fit(y_train, X_train=None)`` / ``.predict(test, X_future=None)``.
"""

from ._device import default_device, resolve_device
from .base import DEFAULT_QUANTILES, BaseForecaster
from .gbm import GBMBase, HGBFeatures, XGBFeatures
from .linear import RidgeFeatures
from .mlp import MLPFeatures
from .seq import RNNFeatures, TCNFeatures
from .stats import SktimeAutoETS, SktimeTheta
from .torch_base import TorchForecaster

__all__ = [
    "DEFAULT_QUANTILES",
    "BaseForecaster",
    "GBMBase",
    "HGBFeatures",
    "MLPFeatures",
    "RNNFeatures",
    "RidgeFeatures",
    "SktimeAutoETS",
    "SktimeTheta",
    "TCNFeatures",
    "TorchForecaster",
    "XGBFeatures",
    "default_device",
    "resolve_device",
]
