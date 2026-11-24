"""Weather features (block "weather") — all delays 0: archived forecasts, as-of.

Input: the national hourly weather frame (unit-suffixed columns, from
``data.validate.aggregate_weather_national``), or raw grid records aggregated
with population weights via :func:`national_weather_frame`.

Derivatives (thresholds/lags calibrated on the observed load-vs-temperature
U-shape and cross-correlation, see notebooks/2_feature_exploration.ipynb):
HDD base 18 °C (load minimum at ~18 °C), CDD base 24 °C, Magnus dew point,
day min/max/amplitude (local day), 3 h pressure gradient (raw pressure is
uninformative alone), weather lags 12/24 h (thermal-inertia peak at ~12 h).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ..data.validate import (
    WEATHER_UNITS,
    WEATHER_VARS_MEAN,
    WEATHER_VARS_MODE,
    aggregate_weather_national,
    validate_weather,
)
from .base import FeatureBlock, as_utc_hourly_index
from .calendar import TZ_PARIS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "assets"
GRID_WEIGHTS_CSV = DATA_DIR / "grid_weights.csv"

HDD_BASE_C = 18.0  # U-shape load minimum observed at ~18 °C
CDD_BASE_C = 24.0
WEATHER_LAGS_H = (12, 24)  # cross-correlation load(t)~T(t-k) peaks at k≈12 h

# raw national columns (unit-suffixed) -> feature names
WX_RENAME = {
    "temperature_2m_c": "wx_t_c",
    "apparent_temperature_c": "wx_apparent_c",
    "relative_humidity_2m_pct": "wx_rh_pct",
    "pressure_msl_hpa": "wx_p_hpa",
    "precipitation_mm": "wx_precip_mm",
    "wind_speed_10m_ms": "wx_wind_ms",
    "shortwave_radiation_wm2": "wx_sw_wm2",
    "cloud_cover_pct": "wx_cloud_pct",
    "is_day_fraction": "wx_is_day",
    "weather_code_code": "wx_code",
}

# Magnus coefficients for the dew point
_MAGNUS_A = 17.62
_MAGNUS_B = 243.12


class GridWeight(BaseModel):
    """One row of the embedded grid-weights CSV (population weights)."""

    model_config = ConfigDict(extra="forbid")

    latitude: float
    longitude: float
    weight: float

    @field_validator("weight")
    @classmethod
    def _nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weight must be >= 0")
        return v


def load_grid_weights(path: Path = GRID_WEIGHTS_CSV) -> pd.DataFrame | None:
    """Embedded population weights per grid point (normalized to sum 1).

    ``None`` when the CSV has not been generated yet (uniform fallback =
    plain grid mean). Generate it with ``scripts/build_grid_weights.py``.
    """
    if not path.exists():
        logger.info("no grid weights at %s — uniform (plain mean) fallback", path)
        return None
    clean: list[dict[str, Any]] = []
    for record in pd.read_csv(path).to_dict("records"):
        try:
            clean.append(GridWeight.model_validate(record).model_dump())
        except ValidationError as exc:
            raise ValueError(f"bad grid weight row: {exc.errors()[0]['msg']}") from exc
    df = pd.DataFrame(clean)
    if df.duplicated(["latitude", "longitude"]).any():
        raise ValueError("duplicate lat/lon in grid weights")
    df["weight"] = df["weight"] / df["weight"].sum()
    return df


def national_weather_frame(
    raw_grid: pd.DataFrame, weights: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Validate raw grid records, then aggregate to national hourly.

    ``weights=None``: plain mean over the grid (current processed behaviour).
    With weights (population per grid point): weighted mean per UTC hour,
    weighted vote for ``weather_code``.
    """
    clean, _ = validate_weather(raw_grid)
    if weights is None:
        return aggregate_weather_national(clean)
    wmap = {
        (round(r.latitude, 4), round(r.longitude, 4)): float(r.weight)
        for r in weights.itertuples()
    }
    sub = clean.copy()
    sub["time"] = pd.to_datetime(sub["time"], utc=True)
    sub["w"] = [
        wmap.get((round(la, 4), round(lo, 4)), 0.0)
        for la, lo in zip(sub["latitude"], sub["longitude"], strict=False)
    ]
    out = pd.DataFrame(
        index=pd.Index(sorted(sub["time"].unique()), name="date_heure_utc")
    )
    for var in WEATHER_VARS_MEAN:
        if var not in sub.columns:
            continue
        num = (sub[var] * sub["w"]).groupby(sub["time"]).sum(min_count=1)
        den = sub["w"].where(sub[var].notna()).groupby(sub["time"]).sum()
        out[f"{var}_{WEATHER_UNITS[var]}"] = num / den
    for var in WEATHER_VARS_MODE:
        if var not in sub.columns:
            continue
        votes = sub.dropna(subset=[var])
        top = (
            votes.sort_values("w", ascending=False)
            .drop_duplicates("time")
            .set_index("time")[var]
        )
        out[f"{var}_{WEATHER_UNITS[var]}"] = top.reindex(out.index)
    return out.sort_index()


def _dew_point_c(t_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Magnus dew point (°C) from temperature and relative humidity (%)."""
    rh = rh_pct.clip(lower=1.0, upper=100.0) / 100.0
    gamma = np.log(rh) + _MAGNUS_A * t_c / (_MAGNUS_B + t_c)
    return _MAGNUS_B * gamma / (_MAGNUS_A - gamma)


def weather_block(wx: pd.DataFrame, *, index: pd.DatetimeIndex) -> FeatureBlock:
    """Weather features on the UTC hourly ``index`` (delays 0: as-of archive).

    Only ``temperature_2m_c`` and ``relative_humidity_2m_pct`` are required
    (HDD/CDD, dew point); the other raw variables are carried through when
    present.
    """
    index = as_utc_hourly_index(index)
    df = wx.reindex(index)
    missing = {"temperature_2m_c", "relative_humidity_2m_pct"} - set(df.columns)
    if missing:
        raise ValueError(f"weather frame missing required columns: {sorted(missing)}")
    renamed = df.rename(columns=WX_RENAME)
    keep = [dst for src, dst in WX_RENAME.items() if src in df.columns]
    frame = renamed[keep].copy()

    t = renamed["wx_t_c"]
    rh = renamed["wx_rh_pct"]
    frame["wx_hdd18"] = (HDD_BASE_C - t).clip(lower=0.0)
    frame["wx_cdd24"] = (t - CDD_BASE_C).clip(lower=0.0)
    frame["wx_dewpoint_c"] = _dew_point_c(t, rh)

    local_day = index.tz_convert(TZ_PARIS).normalize()
    frame["wx_t_min_day"] = t.groupby(local_day).transform("min")
    frame["wx_t_max_day"] = t.groupby(local_day).transform("max")
    frame["wx_t_amp_day"] = frame["wx_t_max_day"] - frame["wx_t_min_day"]

    if "wx_p_hpa" in renamed.columns:
        # pressure gradient: the raw level alone is uninformative
        frame["wx_dp_3h"] = renamed["wx_p_hpa"].diff(3)

    for lag in WEATHER_LAGS_H:
        frame[f"wx_t_lag{lag}h"] = t.shift(lag)
        frame[f"wx_hdd18_lag{lag}h"] = frame["wx_hdd18"].shift(lag)
        frame[f"wx_cdd24_lag{lag}h"] = frame["wx_cdd24"].shift(lag)

    delays = {c: 0 for c in frame.columns}  # archived forecasts: known at as_of
    return FeatureBlock("weather", frame, delays)
