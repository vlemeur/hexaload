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


def test_fit_predict_roundtrip() -> None:
    X, y = make_xy()
    m = RidgeFeatures().fit(X.loc[:"2025-02-01"], y.loc[:"2025-02-01"])
    pred = m.predict(X.loc["2025-02-01":])
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(X.loc["2025-02-01":].index)
    assert pred.notna().all()
    # the model recovers the linear relation: MAE small relative to signal
    mae = float((pred - y.loc["2025-02-01":]).abs().mean())
    assert mae < 5.0


def test_deterministic_and_clipped() -> None:
    X, y = make_xy()
    m1 = RidgeFeatures().fit(X, y)
    m2 = RidgeFeatures().fit(X, y)
    pd.testing.assert_series_equal(m1.predict(X), m2.predict(X))
    Xneg = X.copy()
    Xneg["f1"] = -1e9  # force negative raw prediction
    assert (RidgeFeatures().fit(X, y).predict(Xneg) >= 0).all()


def test_nan_rows_and_missing_columns() -> None:
    X, y = make_xy()
    y2 = y.copy()
    y2.iloc[:100] = np.nan
    m = RidgeFeatures().fit(X, y2)  # imputer handles X NaN; y NaN rows dropped
    Xnan = X.copy()
    Xnan.iloc[0, 0] = np.nan
    assert m.predict(Xnan).notna().all()  # imputed
    with pytest.raises(RuntimeError, match="fit"):
        RidgeFeatures().predict(X)
    with pytest.raises(ValueError, match="missing feature columns"):
        m.predict(X.drop(columns=["f1"]))
