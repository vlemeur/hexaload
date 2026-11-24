"""Model families — one common interface: .fit(X_train, y_train), .predict(X_future)."""

from .linear import RidgeFeatures

__all__ = ["RidgeFeatures"]
