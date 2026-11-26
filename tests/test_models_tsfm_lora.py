"""TimestralLoRA: LoRA injection + per-fold fine-tune — CPU-only, fake pipe.

The pretrained pipeline is faked (no weight download, no GPU probe); the fake
forecast is differentiable through the model so gradients reach the adapters.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from hexaload.models import TimestralLoRA, tsfm_lora
from hexaload.models.tsfm_lora import LoRALinear, inject_lora

LEVELS = (0.1, 0.5, 0.9)


def make_xy(n: int = 24 * 20):
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    y = pd.Series(
        50_000 + 3_000 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 100, n),
        index=idx,
        name="y",
    )
    X = pd.DataFrame(
        {"wx": rng.normal(size=n), "cal": (idx.dayofweek >= 5).astype(float)}, index=idx
    )
    return X, y


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FakePipe:
    patch_size = 8

    def __init__(self) -> None:
        self.model = FakeModel()
        self.device = torch.device("cpu")

    def forecast(self, context, prediction_length, known_covariates=None, **kw):
        h = prediction_length
        base = float(np.nanmean(np.asarray(context, dtype=np.float32)))
        q = np.stack([base + (a - 0.5) * 2000 for a in LEVELS], axis=0)  # (Q,)
        Q = np.repeat(q[None, :], h, axis=0)  # (H, Q)
        return SimpleNamespace(
            quantiles=torch.tensor(Q)[None, None],  # (1, 1, H, Q)
            quantile_levels=LEVELS,
            median=torch.tensor(np.full(h, base))[None, None],
        )


def fake_forecast_grad(pipe, context, prediction_length, known_covariates=None, **kw):
    ctx = torch.as_tensor(np.asarray(context), dtype=torch.float32)  # (B, L)
    z = pipe.model(ctx[:, -4:])  # (B, 3) — differentiable through the adapters
    q = z[:, None, None, :] * torch.tensor(LEVELS, dtype=z.dtype)[None, None, None, :]
    q = q.expand(ctx.shape[0], 1, prediction_length, len(LEVELS))  # (B, 1, H, Q)
    return SimpleNamespace(quantiles=q, quantile_levels=LEVELS)


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> FakePipe:
    pipe = FakePipe()  # shared across fits: exercises the reset logic
    monkeypatch.setattr(
        TimestralLoRA, "_load", lambda self: setattr(self, "_pipe", pipe)
    )
    monkeypatch.setattr(tsfm_lora, "_forecast_grad", fake_forecast_grad)
    tsfm_lora._LORA_INIT.clear()
    return pipe


def tiny(**kw) -> TimestralLoRA:
    kw.setdefault("context_hours", 24 * 10)
    kw.setdefault("n_steps", 5)
    kw.setdefault("batch", 2)
    kw.setdefault("train_context_hours", 32)
    kw.setdefault("train_horizon", 4)
    kw.setdefault("device", "cpu")
    return TimestralLoRA(**kw)


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k
    }


def test_inject_lora_freezes_base() -> None:
    m = FakeModel()
    base_before = m.head.weight.detach().clone()
    assert inject_lora(m, r=2, alpha=4.0) == 1
    assert isinstance(m.head, LoRALinear)
    assert not m.head.base.weight.requires_grad
    assert m.head.lora_A.requires_grad and m.head.lora_B.requires_grad
    assert torch.equal(m.head.weight, base_before)  # .weight property passthrough
    assert m.head(torch.randn(3, 4)).shape == (3, 3)


def test_fit_trains_adapters_and_predicts(patched: FakePipe) -> None:
    X, y = make_xy()
    m = tiny().fit(y, X)
    head = patched.model.head
    assert isinstance(head, LoRALinear)
    assert not head.base.weight.requires_grad
    assert head.lora_B.detach().abs().sum() > 0  # training moved the adapters
    test_idx = y.index[-24:]
    pred = m.predict(test_idx, X.loc[test_idx])
    assert len(pred) == 24
    q = m.predict_quantiles(test_idx, X.loc[test_idx])
    assert q.shape == (24, 19)


def test_adapters_reset_each_fold(patched: FakePipe) -> None:
    X, y = make_xy()
    m1 = tiny().fit(y, X)
    head = patched.model.head
    assert isinstance(head, LoRALinear)
    s1 = lora_state(patched.model)
    base_w = head.base.weight.detach().clone()
    # simulate a previous fold's adapters: perturb, then a new instance (new fold)
    with torch.no_grad():
        head.lora_B.add_(1.0)
    m2 = tiny().fit(y, X)  # same process, same cached pipe -> reset from init
    s2 = lora_state(patched.model)
    assert set(s1) == set(s2)
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k  # deterministic retrain from the init
    assert torch.equal(head.base.weight, base_w)  # base never trains
    assert m2 is not m1
