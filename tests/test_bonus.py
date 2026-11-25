"""Synthetic tests for the bonus bricks (ensemble / Mondrian / EWMA).

CI-safe: no data/processed, no network, no GPU — dummy members only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hexaload.models import DEFAULT_QUANTILES, BaseForecaster
from hexaload.models.conformal import BiasEWMA, MondrianConformal
from hexaload.models.ensemble import EnsembleFeatures
from hexaload.tuning.spaces import VARIANT_MODELS, build_variant


def _series(vals, start="2026-01-01", freq="h"):
    idx = pd.date_range(start, periods=len(vals), freq=freq, tz="UTC")
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


class ConstModel(BaseForecaster):
    """Predicts a constant (per instance), optionally biased per Paris hour."""

    def __init__(self, value=50000.0, hour_bias=None):
        self.value = value
        self.hour_bias = hour_bias or {}

    def fit(self, y_train, X_train=None):
        return self

    def predict(self, test, X_future=None):
        test = pd.DatetimeIndex(test)
        hours = test.tz_convert("Europe/Paris").hour
        vals = self.value + np.array([self.hour_bias.get(h, 0.0) for h in hours])
        return pd.Series(vals, index=test, name="y_pred")

    def predict_quantiles(self, test, X_future=None, quantiles=DEFAULT_QUANTILES):
        pred = self.predict(test, X_future)
        return pd.DataFrame(
            {f"q{a:g}": pred + (a - 0.5) * 1000 for a in quantiles}, index=pred.index
        )


def _X(idx):
    return pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)


def test_ensemble_weights_convex_and_prediction_is_weighted_sum():
    idx = pd.date_range("2026-01-01", periods=24 * 40, freq="h", tz="UTC")
    y = _series(50000 + 100 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi))
    X = _X(idx)
    ens = EnsembleFeatures([ConstModel(50000.0), ConstModel(51000.0)]).fit(y, X)
    w = ens._weights
    assert w is not None and len(w) == 2
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    test = pd.date_range("2026-02-10", periods=24, freq="h", tz="UTC")
    pred = ens.predict(test, _X(test))
    expect = w[0] * 50000.0 + w[1] * 51000.0
    assert np.allclose(pred.to_numpy(), expect)


def test_ensemble_quantiles_are_conformal_and_ordered():
    idx = pd.date_range("2026-01-01", periods=24 * 40, freq="h", tz="UTC")
    y = _series(50000 + np.random.default_rng(0).normal(0, 300, len(idx)))
    ens = EnsembleFeatures([ConstModel(50000.0)]).fit(y, _X(idx))
    test = pd.date_range("2026-02-10", periods=24, freq="h", tz="UTC")
    q = ens.predict_quantiles(test, _X(test))
    cols = [f"q{a:g}" for a in DEFAULT_QUANTILES]
    assert list(q.columns) == cols
    assert (np.diff(q.to_numpy(), axis=1) >= -1e-9).all()  # monotone in level
    assert (q["q0.05"] <= q["q0.5"]).all() and (q["q0.5"] <= q["q0.95"]).all()


def test_mondrian_per_hour_offsets():
    idx = pd.date_range("2026-01-01", periods=24 * 40, freq="h", tz="UTC")
    # actual = 50000 + 2000 at Paris hour 19 (evening peak), else 50000
    hours = idx.tz_convert("Europe/Paris").hour
    y = _series(50000 + np.where(hours == 19, 2000.0, 0.0))
    inner = ConstModel(50000.0)  # misses the peak entirely
    mon = MondrianConformal(inner).fit(y, _X(idx))
    test = pd.date_range("2026-02-10", periods=24, freq="h", tz="UTC")
    q = mon.predict_quantiles(test, _X(test), quantiles=(0.5,))
    th = test.tz_convert("Europe/Paris").hour
    med = q["q0.5"].to_numpy() - 50000.0
    assert med[th == 19].mean() > med[th != 19].mean() + 500  # peak hour wider/higher


def test_bias_ewma_removes_constant_bias():
    idx = pd.date_range("2026-01-01", periods=24 * 40, freq="h", tz="UTC")
    y = _series(np.full(len(idx), 51000.0))
    inner = ConstModel(50000.0)  # constant -1000 bias vs actual
    corr = BiasEWMA(inner, alpha=0.5).fit(y, _X(idx))
    test = pd.date_range("2026-02-10", periods=24, freq="h", tz="UTC")
    pred = corr.predict(test, _X(test))
    assert np.allclose(pred.to_numpy(), 51000.0, atol=1.0)
    q = corr.predict_quantiles(test, _X(test))
    assert np.allclose(q["q0.5"].to_numpy(), 51000.0, atol=1.0)


def test_build_variant_wiring():
    for name in VARIANT_MODELS:
        m = build_variant(name, {})
        assert isinstance(m, BaseForecaster)
    combo = build_variant("hgb_features_ewma_mondrian", {})
    assert isinstance(combo, MondrianConformal)
    assert isinstance(combo._inner, BiasEWMA)
    with pytest.raises(ValueError):
        build_variant("nope", {})
