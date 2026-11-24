"""TSFM wrappers (Timestral / Chronos-Bolt): contract + anti-leakage, offline.

The pretrained pipelines are faked (no weight download in tests).
"""

import numpy as np
import pandas as pd
import pytest

from hexaload.models import ChronosBoltTSFM, TimestralTSFM


def make_y(n: int = 24 * 40) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    y = 50_000 + 3_000 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 100, n)
    return pd.Series(y, index=idx, name="consommation")


class FakeForecast:
    def __init__(self, q: np.ndarray, levels: tuple, med: np.ndarray) -> None:
        import torch

        self.quantiles = torch.tensor(q)[None, None]  # (1,1,H,Q)
        self.quantile_levels = levels
        self.median = torch.tensor(med)[None, None]


class FakeTimestral:
    quantile_levels = tuple(np.round(np.arange(0.01, 1.0, 0.05), 2))  # 20ish levels

    def __init__(self) -> None:
        self.last_known_shape = None

    def forecast(self, context, prediction_length, known_covariates=None, **kw):
        h = prediction_length
        levels = tuple(np.concatenate([[0.01], np.arange(0.05, 1.0, 0.05), [0.99]]))
        base = float(np.nanmean(context))
        q = np.stack([base + (a - 0.5) * 2000 for a in levels], axis=0)  # (Q,)
        Q = np.repeat(q[None, :], h, axis=0)  # (H,Q)
        if known_covariates is not None:
            self.last_known_shape = known_covariates.shape  # (1,C,T+H)
        return FakeForecast(Q, levels, np.full(h, base))


class FakeChronosBolt:
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # training levels

    def predict(self, inputs, prediction_length, **kw):
        import torch

        h = prediction_length
        base = float(np.nanmean(inputs.numpy() if hasattr(inputs, "numpy") else inputs))
        q = np.stack([base + (a - 0.5) * 2000 for a in self.quantiles], axis=0)  # (Q9,)
        P = np.repeat(q[:, None], h, axis=1)  # (Q9, H)
        return torch.tensor(P[None])  # (1, Q9, H)


@pytest.fixture
def patched_tsfm(monkeypatch):
    monkeypatch.setattr(
        TimestralTSFM, "_load", lambda self: setattr(self, "_pipe", FakeTimestral())
    )
    monkeypatch.setattr(
        ChronosBoltTSFM, "_load", lambda self: setattr(self, "_pipe", FakeChronosBolt())
    )


def test_timestral_with_known_covariates(patched_tsfm) -> None:
    y = make_y()
    idx = pd.DatetimeIndex(y.index)
    X = pd.DataFrame(
        {
            "wx": np.sin(np.arange(len(y)) / 24),
            "cal": (idx.dayofweek >= 5).astype(float),
        },
        index=idx,
    )
    train_idx = pd.DatetimeIndex(idx[:-24])
    test_idx = pd.DatetimeIndex(idx[-24:])
    m = TimestralTSFM(context_hours=24 * 14).fit(y.loc[train_idx], X.loc[train_idx])
    pred = m.predict(test_idx, X.loc[test_idx])
    assert pred.index.equals(test_idx)
    assert len(pred) == 24 and pred.notna().all()
    # known_covariates must cover context + horizon: (1, C, T+H)
    assert m._pipe.last_known_shape == (1, 2, 24 * 14 + 24)
    q = m.predict_quantiles(test_idx, X.loc[test_idx], quantiles=(0.05, 0.5, 0.95))
    assert q.shape == (24, 3)
    assert (q["q0.05"] <= q["q0.5"]).all() and (q["q0.5"] <= q["q0.95"]).all()
    # cache: predict then predict_quantiles must not re-run the model
    assert m._cache_test == tuple(test_idx)


def test_chronos_bolt_univariate(patched_tsfm) -> None:
    y = make_y()
    idx = y.index
    train_idx = pd.DatetimeIndex(idx[:-24])
    test_idx = pd.DatetimeIndex(idx[-24:])
    m = ChronosBoltTSFM(context_hours=24 * 14).fit(y.loc[train_idx])
    pred = m.predict(test_idx)
    assert pred.index.equals(test_idx) and len(pred) == 24
    q = m.predict_quantiles(test_idx, quantiles=(0.05, 0.5, 0.95))
    assert q.shape == (24, 3)
    assert (q["q0.05"] <= q["q0.5"]).all()


def test_anti_leakage_context_truncated_at_fold_origin(patched_tsfm) -> None:
    y = make_y()
    idx = pd.DatetimeIndex(y.index)
    train_idx = pd.DatetimeIndex(idx[:-24])
    test_idx = pd.DatetimeIndex(idx[-24:])
    m = TimestralTSFM(context_hours=24 * 14).fit(y.loc[train_idx])
    m.predict(test_idx)
    # context = tail of the TRAIN fold only, never the test block
    ctx = m._ctx
    assert ctx is not None
    assert len(ctx) == 24 * 14
    assert pd.Timestamp(idx[-25]) == pd.Timestamp(train_idx[-1])
    # context values are exactly the last context_hours train values
    np.testing.assert_allclose(ctx, y.loc[train_idx].iloc[-24 * 14 :].to_numpy())


def test_contract(patched_tsfm) -> None:
    y = make_y()
    with pytest.raises(RuntimeError, match="fit"):
        TimestralTSFM().predict(y.index[-24:])
    with pytest.raises(ValueError, match="backend"):
        from hexaload.models.tsfm import TSFMForecaster

        TSFMForecaster("nope", "nope")._load()
