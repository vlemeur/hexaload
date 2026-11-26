"""Zero-shot time-series foundation models (TSFMs): Timestral (Mistral)
+ Chronos-Bolt (Amazon).

Zero-shot: ``fit`` only stores the context — the tail of the train fold,
truncated at the fold origin (anti-leakage: the test block is never in the
context) — plus the exogenous covariates. ``predict`` calls the pretrained
model once per fold.

- **Timestral**: multivariate, probabilistic (21 quantile levels), and takes
  ``known_covariates`` over context+horizon — we pass our exogenous features
  (weather forecasts, calendar) there, known in advance on the whole horizon.
- **Chronos-Bolt**: univariate; quantiles via ``predict_quantiles``.

Weights come from the Hugging Face hub (download once, then HF_HUB_OFFLINE=1
for offline backtests).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._device import resolve_device
from .base import DEFAULT_QUANTILES, BaseForecaster

TIMESTRAL_209M = "mistralai/Timestral-209M"
CHRONOS_BOLT_BASE = "amazon/chronos-bolt-base"


_PIPE_CACHE: dict[tuple[str, str, str], Any] = {}  # (backend, repo_id, device) -> pipe


class TSFMForecaster(BaseForecaster):
    """Common wrapper for zero-shot TSFMs.

    ``backend``: ``"timestral"`` or ``"chronos_bolt"``. ``context_hours``: how
    many trailing train points form the context (capped by the model itself).
    """

    def __init__(
        self,
        backend: str,
        repo_id: str,
        *,
        context_hours: int = 24 * 365,
        device: str | None = None,
    ) -> None:
        self._backend = backend
        self._repo_id = repo_id
        self._context_hours = context_hours
        self._device = resolve_device(device)
        self._pipe: Any = None
        self._ctx: np.ndarray | None = None  # (T,) float32
        self._X_ctx: np.ndarray | None = None  # (T, C) float32 exogenous
        self._cache_test: tuple | None = None
        self._cache_q: np.ndarray | None = None  # (H, Q)
        self._cache_pred: pd.Series | None = None

    def _load(self) -> None:
        """Load the pretrained pipeline once per process (HF disk cache avoids
        re-downloading; the class-level cache avoids re-loading per fold)."""
        key = (self._backend, self._repo_id, self._device)
        if key in _PIPE_CACHE:
            self._pipe = _PIPE_CACHE[key]
            return
        if self._backend == "timestral":
            from timestral import TimestralForecaster

            self._pipe = TimestralForecaster.from_pretrained(
                self._repo_id, device=self._device
            )
        elif self._backend == "chronos_bolt":
            from chronos import ChronosBoltPipeline

            self._pipe = ChronosBoltPipeline.from_pretrained(
                self._repo_id, device_map=self._device
            )
        else:
            raise ValueError(f"unknown TSFM backend {self._backend!r}")
        _PIPE_CACHE[key] = self._pipe

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> TSFMForecaster:
        if self._pipe is None:
            self._load()
        y = y_train.interpolate(method="time").bfill().ffill()
        ctx = y.iloc[-self._context_hours :]
        self._ctx = ctx.to_numpy(dtype=np.float32)
        if X_train is not None:
            self._X_ctx = (
                X_train.reindex(ctx.index).fillna(0.0).to_numpy(dtype=np.float32)
            )
        self._cache_test = None
        return self

    def _forecast_q(
        self, test: pd.Index, X_future: pd.DataFrame | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the TSFM once; return (quantiles (H,Q), median (H,))."""
        if self._pipe is None or self._ctx is None:
            raise RuntimeError("call fit() before predict()")
        test = pd.DatetimeIndex(test)
        h = len(test)
        if self._cache_test == tuple(test) and self._cache_q is not None:
            return self._cache_q, self._cache_pred.to_numpy()  # type: ignore[union-attr]
        if self._backend == "timestral":
            known = None
            if self._X_ctx is not None and X_future is not None:
                x_fut = (
                    X_future.reindex(test)
                    .reindex(columns=self._cov_cols)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )
                known = np.concatenate([self._X_ctx, x_fut], axis=0)[None]  # (1,T+H,C)
                known = np.transpose(known, (0, 2, 1))  # (1,C,T+H)
            fc = self._pipe.forecast(
                self._ctx[None, :],  # (1,T)
                prediction_length=h,
                known_covariates=known,
            )
            levels = np.asarray(fc.quantile_levels, dtype=np.float64)  # (21,)
            Q = fc.quantiles[0, 0].cpu().numpy().astype(np.float64)  # (H,21)
            med = fc.median[0, 0].cpu().numpy().astype(np.float64)  # (H,)
            # interpolate the 21 native levels onto our quantile grid
            target = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
            q = np.stack(
                [np.interp(target, levels, Q[i]) for i in range(h)], axis=0
            )  # (H,19)
        else:  # chronos_bolt: predict -> (1, Q_train, H) with fixed training levels
            import torch

            ctx_t = torch.from_numpy(np.nan_to_num(self._ctx)).float()[None, :]  # (1,T)
            preds = self._pipe.predict(ctx_t, prediction_length=h)  # (1,Q9,H)
            levels = np.asarray(
                [float(v) for v in self._pipe.quantiles]
            )  # (9,) 0.1..0.9
            P = preds[0].cpu().numpy().T.astype(np.float64)  # (H,Q9)
            target = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
            # interpolate the training levels onto our grid (np.interp clamps at edges)
            q = np.stack([np.interp(target, levels, P[i]) for i in range(h)], axis=0)
            i50 = int(np.argmin(np.abs(levels - 0.5)))
            med = P[:, i50]  # median = training level 0.5
        self._cache_test = tuple(test)
        self._cache_q = q
        self._cache_pred = pd.Series(np.clip(med, 0.0, None), index=test, name="y_pred")
        return q, med

    def predict(
        self, test: pd.Index, X_future: pd.DataFrame | None = None
    ) -> pd.Series:
        test = pd.DatetimeIndex(test)
        self._forecast_q(test, X_future)
        assert self._cache_pred is not None
        return self._cache_pred

    def predict_quantiles(
        self,
        test: pd.Index,
        X_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        test = pd.DatetimeIndex(test)
        q, _ = self._forecast_q(test, X_future)  # (H, len(DEFAULT_QUANTILES))
        cols = {
            f"q{a:g}": q[:, k]
            for k, a in enumerate(DEFAULT_QUANTILES)
            if a in set(quantiles)
        }
        return pd.DataFrame(cols, index=test)


class TimestralTSFM(TSFMForecaster):
    """Timestral (Mistral) — with known-future covariates when X is provided."""

    def __init__(self, repo_id: str = TIMESTRAL_209M, **kwargs) -> None:
        super().__init__("timestral", repo_id, **kwargs)
        self._cov_cols: list[str] | None = None

    def _load(self) -> None:
        from timestral import TimestralForecaster

        device = self._device
        if device == "mps":
            # MPS has no float64 kernels and Timestral's scaler casts to float64
            # on the model's device: downcast those casts to float32 on MPS only
            # (weights are float32; CPU path stays bit-exact, validated vs CPU).
            from ._mps_shim import apply_mps_f64_shim

            apply_mps_f64_shim()
        key = (self._backend, self._repo_id, device)
        if key in _PIPE_CACHE:  # load the weights once per process, not per fold
            self._pipe = _PIPE_CACHE[key]
            return
        self._pipe = TimestralForecaster.from_pretrained(self._repo_id, device=device)
        _PIPE_CACHE[key] = self._pipe

    def fit(
        self, y_train: pd.Series, X_train: pd.DataFrame | None = None
    ) -> TimestralTSFM:
        if X_train is not None:
            self._cov_cols = list(X_train.columns)
        return super().fit(y_train, X_train)


class ChronosBoltTSFM(TSFMForecaster):
    """Chronos-Bolt (Amazon) — univariate, quantiles via predict_quantiles."""

    def __init__(self, repo_id: str = CHRONOS_BOLT_BASE, **kwargs) -> None:
        super().__init__("chronos_bolt", repo_id, **kwargs)
