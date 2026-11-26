"""Daily scoring of the published D-1 forecasts against actuals.

Reads ``docs/data/forecasts/<J>.json`` (written by scripts/forecast.py) and the
on-disk actuals (stitched cons-def + tr). For each published day J: MAE and
MASE (sp=24) per model and for RTE's own prevision_j1, plus the CUSUM on the
model-vs-RTE gap. Pure / offline — the scheduled run refreshes the cache
first (forecast.yml).
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from hexaload.metrics import DEFAULT_SP, mae, mase

from .cusum import cusum_series

PARIS = ZoneInfo("Europe/Paris")
WINDOW_H = 24 * 365


def json_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of records with NaN/NaT as None (strict-JSON safe).

    ``df.where(pd.notna(df), None)`` alone does NOT work: on a float64 column
    None is coerced back to NaN, and ``json.dumps`` then emits a bare ``NaN``
    — invalid strict JSON (breaks check-json and the site's JSON.parse).
    The column must be object dtype first so None sticks.
    """
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def day_test_index(date_str: str) -> pd.DatetimeIndex:
    """The 24 h D-1 block of target day ``date_str`` (Paris midnight → UTC)."""
    target = pd.Timestamp(date_str).tz_localize(PARIS).tz_convert("UTC")
    return pd.date_range(target, periods=24, freq="h", tz="UTC")


def daily_scores(
    forecast_dir: Path | str,
    y: pd.Series,
    ref: pd.Series,
    *,
    source: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per published forecast day: per-model MAE/MASE + RTE + CUSUM.

    ``source`` (cons-def/tr mask) marks whether the day's actuals are already
    consolidated (cons-def) or still provisional (tr tail).
    """
    forecast_dir = Path(forecast_dir)
    idx_path = forecast_dir / "index.json"
    if not idx_path.exists():
        return pd.DataFrame()
    dates = sorted(json.loads(idx_path.read_text()).get("dates", []))
    rows: list[dict] = []
    for date_str in dates:
        path = forecast_dir / f"{date_str}.json"
        if not path.exists():
            continue
        j = json.loads(path.read_text())
        test = day_test_index(date_str)
        y_test = y.reindex(test)
        if y_test.isna().all():
            continue  # actuals not available yet
        as_of = pd.Timestamp(j["as_of"])
        y_train = y.loc[:as_of].dropna().iloc[-WINDOW_H:]
        row: dict = {
            "date": date_str,
            "as_of": str(as_of),
            "provisional": bool(
                source is not None and (source.reindex(test) == "tr").any()
            ),
        }
        gap_vs_rte: dict[str, float] = {}
        rte = j.get("rte_d1")
        if rte is not None and any(v is not None for v in rte):
            rte_pred = pd.Series(rte, index=test)
            row["rte_mae"] = mae(y_test, rte_pred)
            row["rte_mase"] = mase(y_test, rte_pred, y_train=y_train, sp=DEFAULT_SP)
        for name, m in (j.get("models") or {}).items():
            pred = pd.Series(m["median"], index=test)
            row[f"{name}__mae"] = mae(y_test, pred)
            row[f"{name}__mase"] = mase(y_test, pred, y_train=y_train, sp=DEFAULT_SP)
            if "rte_mae" in row:
                gap_vs_rte[name] = row[f"{name}__mae"] - row["rte_mae"]
        # CUSUM on the gap of the first model (sorted) vs RTE, in units of the
        # in-control std of the gap over the scored days
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # CUSUM per model on (model MAE − RTE MAE), standardized by the running
    # std of the gap (min 1 MW to avoid blow-ups)
    model_names = sorted(
        {c[: -len("__mae")] for c in df.columns if c.endswith("__mae")}
    )
    for name in model_names:
        gap = df[f"{name}__mae"] - df["rte_mae"]
        sd = max(float(gap.std(ddof=0)) if len(gap) > 1 else 1.0, 1.0)
        _, signed = cusum_series(gap.to_numpy() / sd, k=0.5, h=5.0)
        df[f"{name}__cusum"] = signed
        df[f"{name}__alarm"] = np.abs(signed) > 5.0
    return df
