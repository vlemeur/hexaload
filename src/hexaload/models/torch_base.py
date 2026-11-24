"""Common training infrastructure for hand-rolled PyTorch forecasters.

One series, fixed horizon ``h``, walk-forward folds. Each concrete model is an
``nn.Module`` mapping ``(x_hist (B,T,F), x_fut (B,H,F)) -> (B,H,Q)`` quantile
forecasts, trained with the multi-quantile pinball loss; the point forecast is
the ``q=0.5`` output. Early stopping on the train-tail validation split (same
14-day tail as the GBM family). Device via :func:`resolve_device` (MPS on
Apple Silicon, CPU in CI/tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ._device import resolve_device
from .base import DEFAULT_QUANTILES, BaseForecaster

Y_COL = "__y__"  # the target is always appended as the last feature column


def pinball_loss(
    pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """
    Multi-quantile pinball loss. ``pred``: (B,H,Q), ``y``: (B,H), ``quantiles``: (Q,).
    """
    diff = y.unsqueeze(-1) - pred  # (B,H,Q)
    q = quantiles.view(1, 1, -1)
    return torch.maximum(q * diff, (q - 1.0) * diff).mean()


class _WindowDataset(Dataset):
    """
    Sliding windows over (Xv (N,F), yv (N,)):
    item = (x_hist (T,F), x_fut (H,F), y (H,)).
    """

    def __init__(
        self,
        Xv: np.ndarray,
        yv: np.ndarray,
        input_size: int,
        h: int,
        valid_t: np.ndarray,
    ) -> None:
        self.Xv = Xv
        self.yv = yv
        self.input_size = input_size
        self.h = h
        self.valid_t = valid_t

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.valid_t[i])
        x_hist = torch.from_numpy(self.Xv[t - self.input_size : t])
        x_fut = torch.from_numpy(self.Xv[t : t + self.h])
        y = torch.from_numpy(self.yv[t : t + self.h])
        return x_hist, x_fut, y


class TorchForecaster(BaseForecaster):
    """Base class for hand-rolled torch forecasters on the feature table."""

    def __init__(
        self,
        *,
        h: int = 24,
        input_size: int = 24 * 7,
        max_epochs: int = 60,
        patience: int = 8,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_hours: int = 24 * 14,
        device: str | None = None,
        seed: int = 0,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> None:
        self._h = h
        self._input_size = input_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._batch_size = batch_size
        self._lr = lr
        self._val_hours = val_hours
        self._device = resolve_device(device)
        self._seed = seed
        self._quantiles = tuple(quantiles)
        self._module: nn.Module | None = None
        self._X_tail: np.ndarray | None = None  # last input_size rows of X_eff
        self._cols_fit: list[str] | None = None  # columns of X_eff at fit (incl. __y__)
        self._x_mean: np.ndarray | None = None  # train standardization (per column)
        self._x_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _new_module(self, n_features: int) -> nn.Module:
        """Build the model: (x_hist (B,T,F), x_fut (B,H,F)) -> (B,H,Q)."""
        raise NotImplementedError

    def _prepare(self, y_train: pd.Series, X_train: pd.DataFrame | None):
        """Align y/X, impute X, append the target as last column.
        Returns (X_eff, y_clean)."""
        y = y_train.copy()
        X = pd.DataFrame(index=y.index) if X_train is None else X_train.reindex(y.index)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        X[Y_COL] = y.interpolate(method="time").bfill().ffill()
        y_clean = y.interpolate(method="time").bfill().ffill()
        return X, y_clean

    @staticmethod
    def _valid_positions(yv: np.ndarray, input_size: int, h: int) -> np.ndarray:
        """Positions t where neither the history window nor the target has NaN."""
        nan = np.isnan(yv)
        c = np.concatenate([[0.0], np.cumsum(nan)])
        n = len(yv)
        ts = np.arange(input_size, n - h + 1)
        ok_hist = (c[ts] - c[ts - input_size]) == 0
        ok_tgt = (c[ts + h] - c[ts]) == 0
        return ts[ok_hist & ok_tgt]

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> TorchForecaster:
        torch.manual_seed(self._seed)
        X, y = self._prepare(y_train, X_train)
        # standardize on the train fold (store stats to un-standardize outputs)
        x_mean = X.mean(numeric_only=True).to_numpy(dtype=np.float64)
        x_std = X.std(numeric_only=True).to_numpy(dtype=np.float64)
        x_std = np.where(x_std > 0, x_std, 1.0)
        y_mean = float(y.mean())
        y_std = float(y.std())
        if not np.isfinite(y_std) or y_std <= 0:
            y_std = 1.0
        self._x_mean, self._x_std = x_mean, x_std
        self._y_mean, self._y_std = y_mean, y_std
        Xv = ((X.to_numpy(dtype=np.float64) - x_mean) / x_std).astype(np.float32)
        yv = ((y.to_numpy(dtype=np.float64) - y_mean) / y_std).astype(np.float32)
        valid_t = self._valid_positions(yv, self._input_size, self._h)
        if len(valid_t) < self._val_hours + 24:
            raise ValueError(
                f"not enough valid windows ({len(valid_t)}) for input_size="
                f"{self._input_size}, h={self._h}"
            )
        n_val = min(self._val_hours, max(1, len(valid_t) // 10))
        train_t, val_t = valid_t[:-n_val], valid_t[-n_val:]
        train_ds = _WindowDataset(Xv, yv, self._input_size, self._h, train_t)
        val_ds = _WindowDataset(Xv, yv, self._input_size, self._h, val_t)
        gen = torch.Generator().manual_seed(self._seed)
        train_dl = DataLoader(
            train_ds, batch_size=self._batch_size, shuffle=True, generator=gen
        )
        val_dl = DataLoader(val_ds, batch_size=self._batch_size, shuffle=False)
        qs = torch.tensor(self._quantiles, dtype=torch.float32, device=self._device)
        module = self._new_module(Xv.shape[1]).to(self._device)
        opt = torch.optim.Adam(module.parameters(), lr=self._lr)
        best_val, best_state, bad = float("inf"), None, 0
        for _epoch in range(self._max_epochs):
            module.train()
            for x_hist, x_fut, yb in train_dl:
                x_hist, x_fut, yb = (
                    x_hist.to(self._device),
                    x_fut.to(self._device),
                    yb.to(self._device),
                )
                opt.zero_grad()
                loss = pinball_loss(module(x_hist, x_fut), yb, qs)
                loss.backward()
                opt.step()
            module.eval()
            with torch.no_grad():
                val_loss = float(
                    torch.stack(
                        [
                            pinball_loss(
                                module(xh.to(self._device), xf.to(self._device)),
                                yb.to(self._device),
                                qs,
                            )
                            for xh, xf, yb in val_dl
                        ]
                    ).mean()
                )
            if val_loss < best_val:
                best_val, best_state, bad = val_loss, module.state_dict(), 0
            else:
                bad += 1
                if bad >= self._patience:
                    break
        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()
        self._module = module
        self._X_tail = Xv[-self._input_size :].copy()
        self._cols_fit = list(X.columns)
        return self

    def _forward_q(self, test: pd.Index, X_future: pd.DataFrame | None) -> np.ndarray:
        """Run the model on the last train window + future exog. Returns (H, Q)."""
        module = self._module
        if module is None or self._X_tail is None:
            raise RuntimeError("call fit() before predict()")
        test = pd.DatetimeIndex(test)
        if len(test) != self._h:
            raise ValueError(f"expected a test block of h={self._h}, got {len(test)}")
        if self._cols_fit is None or self._x_mean is None or self._x_std is None:
            raise RuntimeError("call fit() before predict()")
        n_cols = len(self._cols_fit)
        if X_future is None:
            x_fut = np.zeros((1, self._h, n_cols), dtype=np.float32)
        else:
            xf = X_future.reindex(test)
            if Y_COL not in xf.columns:
                xf[Y_COL] = np.nan
            missing = [c for c in self._cols_fit if c not in xf.columns]
            if missing:
                raise ValueError(f"missing feature columns: {missing[:5]}")
            xf = xf.reindex(columns=self._cols_fit)
            x_fut = xf.fillna(0.0).to_numpy(dtype=np.float64)[None, :, :]
            # standardize with the TRAIN stats; y column is unknown in the future
            x_fut = (x_fut - self._x_mean) / self._x_std
            x_fut[..., -1] = 0.0  # masked target column (standardized NaN -> 0)
        x_fut = x_fut.astype(np.float32)
        x_hist = torch.from_numpy(self._X_tail[None, :, :]).to(self._device)
        x_fut_t = torch.from_numpy(x_fut).to(self._device)
        with torch.no_grad():
            out = module(x_hist, x_fut_t)  # (1,H,Q)
        # un-standardize the target
        return (out[0].cpu().numpy() * self._y_std + self._y_mean).astype(np.float64)

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        """Point forecast = q=0.5 output, clipped at 0 MW."""
        test = pd.DatetimeIndex(test)
        q = self._forward_q(test, X_future)
        i50 = self._quantiles.index(0.5) if 0.5 in self._quantiles else len(q) // 2
        pred = np.clip(q[:, i50], 0.0, None)
        return pd.Series(pred, index=test, name="y_pred")

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Native quantile outputs of the pinball head (columns ``q<level>``)."""
        test = pd.DatetimeIndex(test)
        q = self._forward_q(test, X_future)  # (H, Q) in self._quantiles order
        cols = {
            f"q{a:g}": q[:, k]
            for k, a in enumerate(self._quantiles)
            if a in set(quantiles)
        }
        return pd.DataFrame(cols, index=test)
