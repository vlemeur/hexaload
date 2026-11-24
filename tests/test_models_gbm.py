"""HGBFeatures (sklearn HistGradientBoostingRegressor):
contract, determinism (offline)."""

import numpy as np
import pandas as pd
import pytest

from hexaload.models import HGBFeatures


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
    m = HGBFeatures(max_iter=200).fit(y_tr, X_tr)
    pred = m.predict(test_idx, X_te)
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(test_idx)
    assert pred.notna().all()
    # the GBM recovers the synthetic relation: MAE close to the noise floor
    mae = float((pred - y.loc[test_idx]).abs().mean())
    assert mae < 500.0


def test_deterministic_and_clipped() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    p1 = HGBFeatures(max_iter=100).fit(y_tr, X_tr).predict(test_idx, X_te)
    p2 = HGBFeatures(max_iter=100).fit(y_tr, X_tr).predict(test_idx, X_te)
    pd.testing.assert_series_equal(p1, p2)
    Xneg = X_te.copy()
    Xneg["f1"] = -1e9  # force negative raw prediction
    assert (
        HGBFeatures(max_iter=100).fit(y_tr, X_tr).predict(test_idx, Xneg) >= 0
    ).all()


def test_nan_features_native_and_contract() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    Xnan = X_te.copy()
    Xnan.iloc[0, 0] = np.nan  # HGBR handles missing values natively
    m = HGBFeatures(max_iter=50).fit(y_tr, X_tr)
    assert m.predict(test_idx, Xnan).notna().all()
    with pytest.raises(ValueError, match="X_train"):
        HGBFeatures().fit(y_tr)
    with pytest.raises(ValueError, match="X_future"):
        m.predict(test_idx)
    with pytest.raises(RuntimeError, match="fit"):
        HGBFeatures().predict(test_idx, X_te)
    with pytest.raises(ValueError, match="missing feature columns"):
        m.predict(test_idx, X_te.drop(columns=["f1"]))


def test_validation_tail_is_used_and_train_too_short_raises() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    # default val_hours = 14 days: with a short fold, fit must refuse
    short_y, short_X = y.iloc[: 24 * 10], X.iloc[: 24 * 10]
    with pytest.raises(ValueError, match="validation tail"):
        HGBFeatures().fit(short_y, short_X)
    # but it fits and exposes the early-stopped iteration count on a full fold
    m = HGBFeatures(max_iter=200, val_hours=24 * 7).fit(y_tr, X_tr)
    assert m._model is not None
    assert 1 <= m._model.n_iter_ <= 200
    assert m.predict(test_idx, X_te).notna().all()


def test_predict_quantiles_conformal_coverage() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    m = HGBFeatures(max_iter=200).fit(y_tr, X_tr)
    q = m.predict_quantiles(test_idx, X_te, (0.05, 0.5, 0.95))
    assert q.index.equals(pd.DatetimeIndex(test_idx))
    assert (q["q0.05"] <= q["q0.5"]).all()
    assert (q["q0.5"] <= q["q0.95"]).all()
    y_te = y.loc[test_idx]
    inside = ((y_te >= q["q0.05"]) & (y_te <= q["q0.95"])).mean()
    assert inside >= 0.8  # 90% nominal interval, small synthetic sample
