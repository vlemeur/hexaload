"""Timestral + LoRA, fine-tuned per walk-forward fold (train part only).

``TimestralTSFM`` keeps the pretrained weights frozen (zero-shot).
``TimestralLoRA`` injects low-rank adapters into every ``nn.Linear`` of the
frozen base and trains them on sliding windows sampled from the fold's train
part only — the test block is never inside a training window (no temporal
leakage). Adapters are reset to their initial values at every ``fit``, so each
fold starts from pretrained + its own train part, never from a previous fold.

Training signal: pinball loss over the model's native quantile levels, through
``TimestralForecaster.forecast.__wrapped__`` (the public ``forecast`` is
wrapped in ``@torch.inference_mode()``). Predict is unchanged: the adapters
ride on top of the frozen base.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsfm import _PIPE_CACHE, TIMESTRAL_209M, TimestralTSFM

logger = logging.getLogger(__name__)

# initial adapter values per pipe-cache key, so every fold (new instance, same
# cached pipe in one process) restarts from pretrained + fold train only
_LORA_INIT: dict[tuple, dict[str, torch.Tensor]] = {}


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank adapter ``W + (alpha/r) * B @ A``."""

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = alpha / r

    # timestral reads .weight directly on Linear modules (model.py PatchRepresentation)
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.linear(x, self.base.weight, self.base.bias)
            + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        )


def inject_lora(model: nn.Module, r: int, alpha: float) -> int:
    """Replace every ``nn.Linear`` of ``model`` by a :class:`LoRALinear`."""
    targets = [
        (mod, name, child)
        for _, mod in list(model.named_modules())
        for name, child in mod.named_children()
        if isinstance(child, nn.Linear)
    ]
    for mod, name, child in targets:
        setattr(mod, name, LoRALinear(child, r, alpha))
    return len(targets)


def _forecast_grad(pipe, **kwargs):
    """``forecast`` without ``@torch.inference_mode()``
    : gradients reach the adapters."""
    from timestral import TimestralForecaster

    return TimestralForecaster.forecast.__wrapped__(pipe, **kwargs)


class TimestralLoRA(TimestralTSFM):
    """Timestral-209M zero-shot + LoRA fine-tuned on the fold's train part."""

    def __init__(
        self,
        repo_id: str = TIMESTRAL_209M,
        *,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lr: float = 5e-5,
        n_steps: int = 300,
        batch: int = 2,
        train_context_hours: int = 24 * 30,
        train_horizon: int = 24,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(repo_id, **kwargs)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lr = lr
        self.n_steps = n_steps
        self.batch = batch
        self.train_context_hours = train_context_hours
        self.train_horizon = train_horizon
        self.seed = seed

    def _load(self) -> None:
        """Load the pipe under a dedicated cache key: the zero-shot pipe (same
        repo/device) must never be polluted by injected adapters."""
        from timestral import TimestralForecaster

        device = self._device
        if device == "mps":
            from ._mps_shim import apply_mps_f64_shim

            apply_mps_f64_shim()
        key = (self._backend, self._repo_id, device, "lora")
        if key in _PIPE_CACHE:
            self._pipe = _PIPE_CACHE[key]
            return
        self._pipe = TimestralForecaster.from_pretrained(self._repo_id, device=device)
        _PIPE_CACHE[key] = self._pipe

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> TimestralLoRA:
        super().fit(y_train, X_train)  # loads the pipe, stores context + cov cols
        self._finetune(y_train, X_train)
        return self

    def _finetune(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        pipe = self._pipe
        model = pipe.model
        key = (self._backend, self._repo_id, str(pipe.device), "lora")
        if key not in _LORA_INIT:
            n = inject_lora(model, self.lora_r, self.lora_alpha)
            model.to(pipe.device)
            _LORA_INIT[key] = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
                if "lora_" in k
            }
            logger.info(
                "timestral_lora: %d Linear layers adapted (r=%d)", n, self.lora_r
            )
        else:
            # fresh adapters each fold: pretrained base + this fold's train only
            model.load_state_dict(_LORA_INIT[key], strict=False)
        torch.manual_seed(self.seed)
        ps = pipe.patch_size
        y = y_train.interpolate(method="time").bfill().ffill()
        ytr = y.to_numpy(np.float32)
        Xtr = (
            X_train.reindex(y_train.index).fillna(0.0).to_numpy(np.float32)
            if X_train is not None
            else None
        )
        L = (self.train_context_hours // ps) * ps
        H = self.train_horizon
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=self.lr
        )
        model.train()
        n = len(ytr)
        for step in range(self.n_steps):
            starts = rng.integers(L, n - H + 1, size=self.batch)  # s - L >= 0
            ctx_b = np.stack([ytr[s - L : s] for s in starts])  # (B, L)
            fut_b = np.stack([ytr[s : s + H] for s in starts])  # (B, H)
            known_b = (
                np.stack([Xtr[s - L : s + H].T for s in starts])
                if Xtr is not None
                else None
            )  # (B, C, L+H)
            out = _forecast_grad(
                pipe,
                context=ctx_b,
                prediction_length=H,
                known_covariates=known_b,
                symmetric_averaging=False,  # halves compute; training signal only
            )
            q = out.quantiles  # (B, 1, H, Q), grad flows to the adapters
            levels = torch.as_tensor(
                out.quantile_levels, dtype=q.dtype, device=q.device
            )
            yb = torch.as_tensor(
                fut_b[:, None, :, None], dtype=q.dtype, device=q.device
            )
            u = yb - q
            loss = torch.mean(torch.maximum(levels * u, (levels - 1.0) * u))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % max(1, self.n_steps // 10) == 0:
                logger.info(
                    "lora step %d/%d pinball %.1f", step, self.n_steps, float(loss)
                )
        if pipe.device.type == "mps":
            torch.mps.empty_cache()
