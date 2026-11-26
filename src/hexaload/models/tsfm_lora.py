"""Timestral + LoRA: per-fold fine-tune (backtest row) or one published adapter.

``TimestralTSFM`` keeps the pretrained weights frozen (zero-shot).
``TimestralLoRA`` injects low-rank adapters into every ``nn.Linear`` of the
frozen base and trains them with the pinball loss over the native quantiles,
through ``TimestralForecaster.forecast.__wrapped__`` (the public ``forecast``
is wrapped in ``@torch.inference_mode()``).

Two modes:
- backtest row (``timestral_lora_209m``): ``fit`` re-trains the adapters on the
  fold's train part only, resetting them to their init at every fold
  (module-level ``_LORA_INIT``), so no leakage across folds;
- serving: pass ``adapter_repo_id``/``adapter_path`` — ``fit`` loads the
  published adapter and skips training entirely (inference cost unchanged).

``scripts/train_lora.py`` produces the published adapter: multi-year training,
month-stratified windows, validation tail with best-checkpoint restore, push to
Hugging Face.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsfm import _PIPE_CACHE, TIMESTRAL_209M, TimestralTSFM

if TYPE_CHECKING:
    from hexaload.progress import BaseProgress

logger = logging.getLogger(__name__)

# initial adapter values per pipe-cache key, so every fold (new instance, same
# cached pipe in one process) restarts from pretrained + fold train only
_LORA_INIT: dict[tuple, dict[str, torch.Tensor]] = {}

# HF repo hosting the published production adapter (set once it exists)
DEFAULT_ADAPTER_REPO: str | None = None


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


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of the adapter weights only (``lora_`` keys)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_adapter(model: nn.Module, path: str | Path) -> None:
    from safetensors.torch import save_file

    save_file(lora_state_dict(model), str(path))


def load_adapter_file(path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


def _forecast_grad(pipe, **kwargs):
    """``forecast`` without ``@torch.inference_mode()``: grads reach the adapters."""
    from timestral import TimestralForecaster

    return TimestralForecaster.forecast.__wrapped__(pipe, **kwargs)


def _pinball(q: torch.Tensor, levels: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    u = yb - q
    return torch.mean(torch.maximum(levels * u, (levels - 1.0) * u))


def finetune_lora(
    pipe,
    y: pd.Series,
    X: pd.DataFrame | None,
    *,
    lr: float,
    n_steps: int,
    batch: int,
    train_context_hours: int,
    horizon: int,
    seed: int,
    val_tail_hours: int = 0,
    ckpt_every: int = 200,
    stratify_by_month: bool = False,
    progress: BaseProgress | None = None,
) -> dict:
    """Train the injected adapters of ``pipe`` on sliding windows of ``y``.

    Windows are sampled from ``y`` only — callers pass the train part (anti-
    leakage). If ``val_tail_hours`` > 0, the last ``val_tail_hours`` of ``y``
    are excluded from training starts and used to checkpoint the best adapter
    (restored at the end). Returns ``{"losses", "val", "best_step"}``.
    """
    model = pipe.model
    ps = pipe.patch_size
    y = y.interpolate(method="time").bfill().ffill()
    ytr = y.to_numpy(np.float32)
    Xtr = X.reindex(y.index).fillna(0.0).to_numpy(np.float32) if X is not None else None
    L = (train_context_hours // ps) * ps
    H = horizon
    n = len(ytr)
    n_train = n - val_tail_hours
    if n_train < L + H + 1:
        raise ValueError(f"not enough train data: {n_train} < L({L}) + H({H}) + 1")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()

    def sample_starts(size: int) -> np.ndarray:
        lo, hi = L, n_train - H + 1  # starts in [lo, hi)
        if not stratify_by_month:
            return rng.integers(lo, hi, size=size)
        months = pd.PeriodIndex(y.index[lo:hi], freq="M")
        out = np.empty(size, dtype=np.int64)
        for i in range(size):
            m = months[rng.integers(0, len(months))]
            cand = np.flatnonzero(months == m) + lo
            out[i] = cand[rng.integers(0, len(cand))]
        return out

    def batch_loss(starts: np.ndarray) -> torch.Tensor:
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
        levels = torch.as_tensor(out.quantile_levels, dtype=q.dtype, device=q.device)
        yb = torch.as_tensor(fut_b[:, None, :, None], dtype=q.dtype, device=q.device)
        return _pinball(q, levels, yb)

    val_starts = None
    if val_tail_hours > 0:
        lo = max(L, n - val_tail_hours)
        val_starts = np.linspace(lo, n - H, num=min(32, n - H - lo), endpoint=False)
        val_starts = val_starts.astype(np.int64)

    losses: list[float] = []
    val_hist: list[tuple[int, float]] = []
    best_step, best_val, best_state = -1, float("inf"), None
    for step in range(n_steps):
        loss = batch_loss(sample_starts(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if progress is not None:
            progress.advance(1, loss=f"{losses[-1]:.1f}")
        elif step % max(1, n_steps // 10) == 0:
            logger.info("lora step %d/%d pinball %.1f", step, n_steps, losses[-1])
        if val_starts is not None and (step % ckpt_every == 0 or step == n_steps - 1):
            with torch.no_grad():
                vloss = float(batch_loss(val_starts))
            val_hist.append((step, vloss))
            logger.info("lora step %d/%d val pinball %.1f", step, n_steps, vloss)
            if vloss < best_val:
                best_step, best_val, best_state = step, vloss, lora_state_dict(model)
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        logger.info("restored best adapter (step %d, val %.1f)", best_step, best_val)
    if pipe.device.type == "mps":
        torch.mps.empty_cache()
    return {"losses": losses, "val": val_hist, "best_step": best_step}


class TimestralLoRA(TimestralTSFM):
    """Timestral zero-shot + LoRA, either re-trained per fold or served from a
    published adapter (``adapter_repo_id``/``adapter_path`` → no training)."""

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
        adapter_repo_id: str | None = None,
        adapter_revision: str | None = None,
        adapter_path: str | None = None,
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
        self.adapter_repo_id = adapter_repo_id or DEFAULT_ADAPTER_REPO
        self.adapter_revision = adapter_revision
        self.adapter_path = adapter_path

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

    @property
    def _adapter_source(self) -> str | None:
        if self.adapter_path is not None:
            return str(self.adapter_path)
        if self.adapter_repo_id is not None:
            rev = self.adapter_revision or "main"
            return f"{self.adapter_repo_id}@{rev}"
        return None

    def _ensure_injected(self) -> None:
        model = self._pipe.model
        if not any("lora_" in k for k in model.state_dict()):
            n = inject_lora(model, self.lora_r, self.lora_alpha)
            model.to(self._pipe.device)
            logger.info(
                "timestral_lora: %d Linear layers adapted (r=%d)", n, self.lora_r
            )

    def _load_published_adapter(self) -> None:
        self._ensure_injected()
        if self.adapter_path is not None:
            state = load_adapter_file(self.adapter_path)
        else:
            from huggingface_hub import snapshot_download

            d = snapshot_download(
                self.adapter_repo_id,
                revision=self.adapter_revision,
                allow_patterns=["adapter.safetensors", "adapter_config.json"],
            )
            state = load_adapter_file(Path(d) / "adapter.safetensors")
        if not any("lora_" in k for k in state):
            raise ValueError(
                f"adapter from {self._adapter_source} has no lora_ weights"
            )
        self._pipe.model.load_state_dict(state, strict=False)
        logger.info(
            "timestral_lora: adapter loaded from %s — training skipped",
            self._adapter_source,
        )

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> TimestralLoRA:
        super().fit(y_train, X_train)  # loads the pipe, stores context + cov cols
        if self._adapter_source is not None:
            self._load_published_adapter()
        else:
            self._finetune(y_train, X_train)
        return self

    def _finetune(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        pipe = self._pipe
        model = pipe.model
        key = (self._backend, self._repo_id, str(pipe.device), "lora")
        self._ensure_injected()
        if key not in _LORA_INIT:
            _LORA_INIT[key] = lora_state_dict(model)
        else:
            # fresh adapters each fold: pretrained base + this fold's train only
            model.load_state_dict(_LORA_INIT[key], strict=False)
        finetune_lora(
            pipe,
            y_train,
            X_train,
            lr=self.lr,
            n_steps=self.n_steps,
            batch=self.batch,
            train_context_hours=self.train_context_hours,
            horizon=self.train_horizon,
            seed=self.seed,
        )
