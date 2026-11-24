"""Forecast metrics.

Point metrics (MAE, RMSE, MASE) take aligned ``pd.Series``.

- MASE is imported from sktime (``MeanAbsoluteScaledError``), not
  re-implemented; it needs the in-sample series ``y_train`` for the seasonal
  naive scaling factor.
- CRPS is implemented here on purpose (no gluonts dependency): the fair
  empirical estimator for an ensemble of samples, Jordan et al. (2019),
  https://arxiv.org/abs/1905.03527:

      CRPS(F, y) ~= mean_t [ (1/m) sum_i |s_it - y_t|
                             - (1 / (2 m^2)) sum_i sum_j |s_it - s_jt| ]

  where s_it is member i of the ensemble at time t. For m = 1 it reduces to
  the absolute error |s - y|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

DEFAULT_SP = 24  # hourly series: daily seasonality


def _aligned_pair(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align on the common index and drop NaN pairs."""
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1)
    df = df.dropna()
    if df.empty:
        raise ValueError("no overlapping non-NaN values between y_true and y_pred")
    # strip names: sktime aligns y_true/y_pred on their column index
    return df["y_true"].rename(None), df["y_pred"].rename(None)


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error."""
    t, p = _aligned_pair(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error."""
    t, p = _aligned_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mase(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    *,
    sp: int = DEFAULT_SP,
) -> float:
    """MASE via sktime, scaled by the in-sample seasonal-naive MAE of y_train.

    Values are passed to sktime as plain ndarrays: with pandas objects sktime
    aligns on column/index names and silently misaligns nameless series.
    NaNs are dropped from y_train (the scale must stay finite).
    """
    t, p = _aligned_pair(y_true, y_pred)
    y_train_clean = y_train.dropna().to_numpy(dtype=float)
    if y_train_clean.size <= sp:
        raise ValueError(f"y_train too short for sp={sp} after dropping NaNs")
    metric = MeanAbsoluteScaledError(sp=sp)
    return float(metric(t.to_numpy(), p.to_numpy(), y_train=y_train_clean))


def crps(y_true: pd.Series, samples: pd.DataFrame) -> float:
    """Fair empirical CRPS of an ensemble (rows = time, columns = members)."""
    df = samples.copy()
    df["__y__"] = y_true
    df = df.dropna()
    if df.empty:
        raise ValueError("no overlapping non-NaN values between y_true and samples")
    y = df.pop("__y__").to_numpy(dtype=float)
    s = df.to_numpy(dtype=float)
    m = s.shape[1]
    term1 = np.mean(np.abs(s - y[:, None]))
    pairwise = np.abs(s[:, :, None] - s[:, None, :]).sum(axis=(1, 2)).mean()
    return float(term1 - pairwise / (2.0 * m * m))


def quantile_frame(
    pred: pd.Series, resid: np.ndarray, quantiles: tuple[float, ...]
) -> pd.DataFrame:
    """Split-conformal quantile frame around a point forecast.

    ``resid`` are calibration residuals ``y - pred`` computed on the TRAIN
    fold only (anti-leakage). Returns a DataFrame indexed like ``pred`` with
    one column per level (``q0.05``, ``q0.1``, ...): ``pred + quantile(resid)``.
    """
    r = np.asarray(resid, dtype=float)
    if r.size == 0:
        raise ValueError("no calibration residuals")
    levels = np.quantile(r, list(quantiles), method="linear")
    return pd.DataFrame(
        {
            f"q{a:g}": pred.to_numpy() + v
            for a, v in zip(quantiles, levels, strict=True)
        },
        index=pred.index,
    )


def sample_quantiles(
    qdf: pd.DataFrame,
    quantiles: tuple[float, ...],
    n_samples: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample from a quantile forecast by inverse-CDF interpolation.

    ``qdf``: rows = time, columns = quantile levels in ``quantiles`` order
    (names ``q<level>``). Returns a DataFrame rows = time, columns = samples
    (uniform draws interpolated between the quantile levels, clamped at the
    edges) — the input of :func:`crps`.
    """
    alphas = np.asarray(list(quantiles), dtype=float)
    Q = qdf.to_numpy(dtype=float)
    if Q.shape[1] != alphas.size:
        raise ValueError(
            f"qdf has {Q.shape[1]} columns but {alphas.size} quantile levels given"
        )
    rng = np.random.default_rng(seed)
    U = rng.random((Q.shape[0], n_samples))
    out = np.empty_like(U)
    for i in range(Q.shape[0]):
        out[i] = np.interp(U[i], alphas, Q[i])
    return pd.DataFrame(out, index=qdf.index)
