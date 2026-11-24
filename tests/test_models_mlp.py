"""MLPFeatures (pure PyTorch): contract, determinism, native quantiles. CPU only."""

import numpy as np
import pandas as pd
import pytest

from hexaload.models import MLPFeatures


def make_xy(n: int = 24 * 60):
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    hour = idx.hour.to_numpy()
    f1 = np.sin(2 * np.pi * hour / 24)  # exogenous, correlated with y
    X = pd.DataFrame({"f1": f1, "f2": rng.normal(size=n)}, index=idx)
    y = pd.Series(50_000.0 + 5_000.0 * f1 + rng.normal(0, 200, n), index=idx, name="y")
    return X, y


def split(X: pd.DataFrame, y: pd.Series, cut: str = "2025-02-15", h: int = 24):
    tr = np.asarray(X.index <= pd.Timestamp(cut, tz="UTC"), dtype=bool)
    train_idx = pd.DatetimeIndex(X.index[tr])
    test_idx = pd.DatetimeIndex(X.index[~tr][-h:])  # exactly one horizon block
    return y.loc[train_idx], X.loc[train_idx], test_idx, X.loc[test_idx]


def tiny(**kw) -> MLPFeatures:
    kw.setdefault("h", 24)
    kw.setdefault("input_size", 48)
    kw.setdefault("max_epochs", 40)
    kw.setdefault("patience", 6)
    kw.setdefault("batch_size", 64)
    kw.setdefault("device", "cpu")
    kw.setdefault("hidden", (64, 32))
    return MLPFeatures(**kw)


def test_fit_predict_roundtrip() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    m = tiny().fit(y_tr, X_tr)
    pred = m.predict(test_idx, X_te)
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(test_idx)
    assert pred.notna().all()
    assert (pred >= 0).all()
    mae = float((pred - y.loc[test_idx]).abs().mean())
    assert mae < 1500.0  # signal amplitude is 5000, noise 200


def test_deterministic_with_seed() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    p1 = tiny(seed=7).fit(y_tr, X_tr).predict(test_idx, X_te)
    p2 = tiny(seed=7).fit(y_tr, X_tr).predict(test_idx, X_te)
    pd.testing.assert_series_equal(p1, p2)


def test_predict_quantiles_native_and_ordered() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    m = tiny().fit(y_tr, X_tr)
    q = m.predict_quantiles(test_idx, X_te, quantiles=(0.05, 0.5, 0.95))
    assert q.shape == (24, 3)
    assert list(q.columns) == ["q0.05", "q0.5", "q0.95"]
    assert q.index.equals(test_idx)
    assert (q["q0.05"] <= q["q0.5"]).all()
    assert (q["q0.5"] <= q["q0.95"]).all()
    # median quantile == point forecast
    pd.testing.assert_series_equal(
        q["q0.5"], m.predict(test_idx, X_te), check_names=False
    )


def test_works_without_exog() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, _ = split(X, y)
    m = tiny().fit(y_tr)  # X_train=None -> univariate on y
    pred = m.predict(test_idx)
    assert pred.index.equals(test_idx)
    assert pred.notna().all()


def test_num_workers_and_amp_flag_accepted_on_cpu() -> None:
    # num_workers>0 prefetches batches; use_amp is a no-op on CPU (float32 kept)
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    model = tiny(max_epochs=2, patience=2, num_workers=2, use_amp=True)
    model.fit(y_tr, X_tr)
    pred = model.predict(test_idx, X_te)
    assert len(pred) == 24


def test_contract() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    with pytest.raises(RuntimeError, match="fit"):
        tiny().predict(test_idx, X_te)
    m = tiny().fit(y_tr, X_tr)
    with pytest.raises(ValueError, match="h=24"):
        m.predict(test_idx[:-1], X_te.iloc[:-1])  # block != h
    with pytest.raises(ValueError, match="columns"):
        m.predict(test_idx, X_te.iloc[:, :1])  # wrong feature count
