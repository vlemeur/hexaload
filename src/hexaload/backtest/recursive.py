"""Recursive (iterative) prediction for feature models at a production freeze.

In production the freeze is J-1 12:00 Paris while the D-1 target day starts at
00:00 Paris — a 12 h gap. Load-memory features have delay >= 24 h, so for the
second half of the target day every lag column is NaN at the freeze and a
one-shot ``predict`` collapses to a flat curve (the "afternoon plateau").

Here each hour of the contiguous block is predicted in order, rebuilding the
lag features from the *extended* series: actuals up to ``as_of``, the model's
own predictions beyond. Lags reading past the freeze are thus filled with
earlier predictions instead of NaN. Leakage-safe: every prediction depends only
on information available at ``as_of`` (the chain bottoms out on actuals).
``prevision_j1`` is masked beyond ``as_of + horizon`` (its declared delay), so
RTE's target-day forecast never leaks in when it was published after the
freeze.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from hexaload.features.table import build_feature_table

if TYPE_CHECKING:
    from hexaload.models.base import BaseForecaster

# longest rolling window (720 h) ending at t-horizon, plus margin
PAD_H = 24 * 31


def recursive_predict(
    model: BaseForecaster,
    *,
    y: pd.Series,
    wx: pd.DataFrame,
    prevision_j1: pd.Series | None,
    as_of: pd.Timestamp,
    block: pd.DatetimeIndex,
    columns: list[str],
    horizon: int = 24,
    quantiles: tuple[float, ...] | None = None,
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Predict ``block`` hour by hour, feeding earlier predictions back as lags.

    Returns ``(median, quantile_frame)`` indexed by ``block``
    (``quantile_frame`` is None when ``quantiles`` is None). ``columns`` is the
    training feature order.
    """
    block = pd.DatetimeIndex(block)
    as_of = pd.Timestamp(as_of)
    y_ext = y.loc[:as_of].copy()
    ref = prevision_j1
    if ref is not None:
        # prevision_j1(t) is published ~horizon ahead: known at as_of iff
        # t - horizon <= as_of
        ref = ref.where(ref.index <= as_of + pd.Timedelta(hours=horizon))
    cols = list(columns)
    med: dict[pd.Timestamp, float] = {}
    qcols = [f"q{q:g}" for q in quantiles] if quantiles else None
    qdata: dict[str, dict[pd.Timestamp, float]] = {c: {} for c in qcols or []}
    for t in block:
        idx = pd.date_range(
            t - pd.Timedelta(hours=PAD_H), periods=PAD_H + 1, freq="h", tz="UTC"
        )
        X_t, _man = build_feature_table(
            idx, as_of=None, wx=wx, y=y_ext, prevision_j1=ref, horizon=horizon
        )
        row = X_t.loc[[t], cols]
        # At a production freeze (12:00 Paris J-1), RTE's prevision_j1 for the
        # target day is published only later on J-1, so lag_rte_prevision_j1 is
        # NaN on the second half of the target day — and it is by far the
        # dominant feature for the GBMs. Proxy it with the 24 h load lag
        # (persistence, i.e. roughly what RTE forecasts anyway).
        if "lag_rte_prevision_j1" in row.columns and "lag_load_24h" in row.columns:
            row = row.copy()
            row["lag_rte_prevision_j1"] = row["lag_rte_prevision_j1"].fillna(
                row["lag_load_24h"]
            )
        one = pd.DatetimeIndex([t])
        p = float(model.predict(one, row).iloc[0])
        med[t] = p
        if qcols is not None and quantiles is not None:
            qf = model.predict_quantiles(one, row, quantiles)
            for c in qcols:
                qdata[c][t] = float(qf.iloc[0][c])
        y_ext.loc[t] = p
    median = pd.Series(med, name="y_pred").sort_index()
    qframe = (
        pd.DataFrame({c: pd.Series(v) for c, v in qdata.items()}).sort_index()
        if qcols
        else None
    )
    return median, qframe
