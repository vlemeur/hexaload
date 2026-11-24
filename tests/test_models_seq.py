"""RNN (LSTM/GRU) and TCN pure-PyTorch models: contract, determinism, quantiles.

CPU only.
"""

import numpy as np
import pandas as pd
import pytest

from hexaload.models import RNNFeatures, TCNFeatures


def make_xy(n: int = 24 * 60):
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    f1 = np.sin(2 * np.pi * idx.hour.to_numpy() / 24)
    X = pd.DataFrame({"f1": f1, "f2": rng.normal(size=n)}, index=idx)
    y = pd.Series(50_000.0 + 5_000.0 * f1 + rng.normal(0, 200, n), index=idx, name="y")
    return X, y


def split(X: pd.DataFrame, y: pd.Series, cut: str = "2025-02-15", h: int = 24):
    tr = np.asarray(X.index <= pd.Timestamp(cut, tz="UTC"), dtype=bool)
    train_idx = pd.DatetimeIndex(X.index[tr])
    test_idx = pd.DatetimeIndex(X.index[~tr][-h:])
    return y.loc[train_idx], X.loc[train_idx], test_idx, X.loc[test_idx]


def tiny(cls, **kw):
    kw.setdefault("h", 24)
    kw.setdefault("input_size", 48)
    kw.setdefault("max_epochs", 40)
    kw.setdefault("patience", 6)
    kw.setdefault("batch_size", 64)
    kw.setdefault("device", "cpu")
    return cls(**kw)


@pytest.mark.parametrize(
    "cls,kw",
    [
        (RNNFeatures, {"rnn": "LSTM", "hidden": 32, "num_layers": 1}),
        (RNNFeatures, {"rnn": "GRU", "hidden": 32, "num_layers": 1}),
        (TCNFeatures, {"hidden": 32, "n_blocks": 3, "kernel": 3}),
    ],
)
def test_fit_predict_roundtrip(cls, kw) -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    m = tiny(cls, **kw).fit(y_tr, X_tr)
    pred = m.predict(test_idx, X_te)
    assert pred.index.equals(test_idx)
    assert pred.notna().all() and (pred >= 0).all()
    mae = float((pred - y.loc[test_idx]).abs().mean())
    assert mae < 2000.0  # signal amplitude 5000, noise 200


@pytest.mark.parametrize("cls", [RNNFeatures, TCNFeatures])
def test_deterministic_and_quantiles(cls) -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    kw = {"hidden": 32, "num_layers": 1} if cls is RNNFeatures else {"hidden": 32}
    m1 = tiny(cls, seed=3, **kw).fit(y_tr, X_tr)
    m2 = tiny(cls, seed=3, **kw).fit(y_tr, X_tr)
    p1 = m1.predict(test_idx, X_te)
    p2 = m2.predict(test_idx, X_te)
    pd.testing.assert_series_equal(p1, p2)
    q = m1.predict_quantiles(test_idx, X_te, quantiles=(0.05, 0.5, 0.95))
    assert q.shape == (24, 3)
    assert (q["q0.05"] <= q["q0.5"]).all() and (q["q0.5"] <= q["q0.95"]).all()
    pd.testing.assert_series_equal(q["q0.5"], p1, check_names=False)


def test_contract() -> None:
    X, y = make_xy()
    y_tr, X_tr, test_idx, X_te = split(X, y)
    with pytest.raises(RuntimeError, match="fit"):
        tiny(TCNFeatures, hidden=16).predict(test_idx, X_te)
    m = tiny(TCNFeatures, hidden=16).fit(y_tr, X_tr)
    with pytest.raises(ValueError, match="h=24"):
        m.predict(test_idx[:-1], X_te.iloc[:-1])
