"""XGBFeatures (xgboost): contract, determinism, conformal quantiles (offline)."""

import numpy as np
import pandas as pd
import pytest

xgboost = pytest.importorskip("xgboost")

from hexaload.models import XGBFeatures  # noqa: E402


def make_xy(n: int = 24 * 60) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    f1 = rng.normal(size=n)
    f2 = np.sin(np.arange(n) / 24 * 2 * np.pi)
    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": rng.normal(size=n)}, index=idx)
    y = pd.Series(
        50_000.0 + 3_000.0 * f2 + 500.0 * f1 + rng.normal(0, 50, n), index=idx, name="y"
    )
    return X, y


def split(X: pd.DataFrame, y: pd.Series, cut: str = "2025-02-15"):
    tr = np.asarray(X.index <= pd.Timestamp(cut, tz="UTC"), dtype=bool)
    train_idx = pd.DatetimeIndex(X.index[tr])
    test_idx = pd.DatetimeIndex(X.index[~tr])
    return y.loc[train_idx], X.loc[train_idx], test_idx, X.loc[test_idx]


def test_fit_predict_roundtrip() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    m = XGBFeatures(n_estimators=300).fit(y_tr, X_tr)
    pred = m.predict(test_idx, X_te)
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(test_idx)
    assert pred.notna().all()
    mae = float((pred - y.loc[test_idx]).abs().mean())
    assert mae < 500.0  # recovers the synthetic relation


def test_deterministic_and_clipped() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    p1 = XGBFeatures(n_estimators=100).fit(y_tr, X_tr).predict(test_idx, X_te)
    p2 = XGBFeatures(n_estimators=100).fit(y_tr, X_tr).predict(test_idx, X_te)
    pd.testing.assert_series_equal(p1, p2)
    Xneg = X_te.copy()
    Xneg["f1"] = -1e9
    assert (
        XGBFeatures(n_estimators=100).fit(y_tr, X_tr).predict(test_idx, Xneg) >= 0
    ).all()


def test_nan_features_native_and_contract() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    Xnan = X_te.copy()
    Xnan.iloc[0, 0] = np.nan  # XGBoost handles missing natively (sparsity-aware)
    m = XGBFeatures(n_estimators=50).fit(y_tr, X_tr)
    assert m.predict(test_idx, Xnan).notna().all()
    with pytest.raises(ValueError, match="X_train"):
        XGBFeatures().fit(y_tr)
    with pytest.raises(ValueError, match="X_future"):
        m.predict(test_idx)
    with pytest.raises(RuntimeError, match="fit"):
        XGBFeatures().predict(test_idx, X_te)
    with pytest.raises(ValueError, match="missing feature columns"):
        m.predict(test_idx, X_te.drop(columns=["f1"]))


def test_early_stopping_uses_best_iteration() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    m = XGBFeatures(n_estimators=300, val_hours=24 * 7).fit(y_tr, X_tr)
    assert m._model is not None
    best = m._model.best_iteration
    assert best is not None and best < 300  # stopped early on this easy signal
    assert m.predict(test_idx, X_te).notna().all()


def test_predict_quantiles_conformal_coverage() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    m = XGBFeatures(n_estimators=200).fit(y_tr, X_tr)
    q = m.predict_quantiles(test_idx, X_te, (0.05, 0.5, 0.95))
    assert q.index.equals(pd.DatetimeIndex(test_idx))
    assert (q["q0.05"] <= q["q0.5"]).all()
    assert (q["q0.5"] <= q["q0.95"]).all()
    y_te = y.loc[test_idx]
    inside = ((y_te >= q["q0.05"]) & (y_te <= q["q0.95"])).mean()
    assert inside >= 0.8
