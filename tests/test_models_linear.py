"""RidgeFeatures model (sklearn): fit/predict contract, determinism (offline)."""

import numpy as np
import pandas as pd
import pytest

from hexaload.models import RidgeFeatures


def make_xy(n: int = 24 * 60) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = np.sin(np.arange(n) / 24 * 2 * np.pi)
    X = pd.DataFrame({"f1": x1, "f2": x2, "f3": rng.normal(size=n)}, index=idx)
    y = pd.Series(1000.0 + 50.0 * x1 + 20.0 * x2, index=idx, name="y")
    return X, y


def split(X: pd.DataFrame, y: pd.Series, cut: str = "2025-02-01"):
    tr = np.asarray(X.index <= pd.Timestamp(cut, tz="UTC"), dtype=bool)
    train_idx = pd.DatetimeIndex(X.index[tr])
    test_idx = pd.DatetimeIndex(X.index[~tr])
    return y.loc[train_idx], X.loc[train_idx], test_idx, X.loc[test_idx]


def test_fit_predict_roundtrip() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    m = RidgeFeatures().fit(y_tr, X_tr)
    pred = m.predict(test_idx, X_te)
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(test_idx)
    assert pred.notna().all()
    # the model recovers the linear relation: MAE small relative to signal
    mae = float((pred - y.loc[test_idx]).abs().mean())
    assert mae < 5.0


def test_deterministic_and_clipped() -> None:
    X, y = make_xy()
    m1 = RidgeFeatures().fit(y, X)
    m2 = RidgeFeatures().fit(y, X)
    pd.testing.assert_series_equal(m1.predict(X.index, X), m2.predict(X.index, X))
    Xneg = X.copy()
    Xneg["f1"] = -1e9  # force negative raw prediction
    assert (RidgeFeatures().fit(y, X).predict(X.index, Xneg) >= 0).all()


def test_nan_rows_and_missing_columns() -> None:
    X, y = make_xy()
    y2 = y.copy()
    y2.iloc[:100] = np.nan
    m = RidgeFeatures().fit(y2, X)  # imputer handles X NaN; y NaN rows dropped
    Xnan = X.copy()
    Xnan.iloc[0, 0] = np.nan
    assert m.predict(X.index, Xnan).notna().all()  # imputed
    with pytest.raises(RuntimeError, match="fit"):
        RidgeFeatures().predict(X.index, X)
    with pytest.raises(ValueError, match="X_train"):
        RidgeFeatures().fit(y)
    with pytest.raises(ValueError, match="X_future"):
        m.predict(X.index)
    with pytest.raises(ValueError, match="missing feature columns"):
        m.predict(X.index, X.drop(columns=["f1"]))
