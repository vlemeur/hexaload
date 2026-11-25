"""Model families — one common interface.

``.fit(y_train, X_train=None)`` / ``.predict(test, X_future=None)``.
"""

from ._device import default_device, resolve_device
from .base import DEFAULT_QUANTILES, BaseForecaster
from .conformal import BiasEWMA, MondrianConformal
from .ensemble import EnsembleFeatures
from .gbm import GBMBase, HGBFeatures, XGBFeatures
from .linear import RidgeFeatures
from .mlp import MLPFeatures
from .nhits import NHITS
from .patchtst import PatchTSTFeatures
from .seq import RNNFeatures, TCNFeatures
from .stats import SktimeAutoETS, SktimeTheta
from .torch_base import TorchForecaster
from .tsfm import ChronosBoltTSFM, TimestralTSFM, TSFMForecaster

__all__ = [
    "DEFAULT_QUANTILES",
    "BaseForecaster",
    "BiasEWMA",
    "EnsembleFeatures",
    "MondrianConformal",
    "GBMBase",
    "HGBFeatures",
    "MLPFeatures",
    "NHITS",
    "PatchTSTFeatures",
    "RNNFeatures",
    "RidgeFeatures",
    "SktimeAutoETS",
    "SktimeTheta",
    "TCNFeatures",
    "ChronosBoltTSFM",
    "TimestralTSFM",
    "TSFMForecaster",
    "TorchForecaster",
    "XGBFeatures",
    "default_device",
    "resolve_device",
]
