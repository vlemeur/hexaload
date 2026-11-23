"""Network-free validation and transformation of raw fetched records.

Tolerant on input (stringly-typed numbers, nulls, ISO-8601 timestamps),
STRICT pydantic v2 schema on output. Never touches the network: feed it a
DataFrame of raw records (e.g. from ``fetch.read_cache``) and it returns
clean, typed data plus a validation report.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

logger = logging.getLogger(__name__)

DEFAULT_PERIMETRE = "France"
# Union of both natures for continuity (phase-2 audit): "Données définitives"
# covers 2012→2025-12, "Données consolidées" covers 2025-12→2026-08-31 (the
# cons-def feed then stops on ODRE; see stitch.py for the real-time tail).
DEFAULT_NATURES = ("Données définitives", "Données consolidées")
# Nature of the eco2mix "*-tr" (temps réel) datasets: same indicator, provisional.
NATURE_TR = "Données temps réel"


class Eco2mixRecord(BaseModel):
    """One raw eco2mix record, strictly typed after input coercion.

    Only the fields the benchmark uses are declared strictly; any extra
    columns served by the API are kept as-is (``extra="allow"``).
    """

    model_config = ConfigDict(extra="allow")

    date_heure: datetime
    date: str
    heure: str
    perimetre: str
    nature: str
    consommation: int | None
    prevision_j1: int | None
    prevision_j: int | None

    @field_validator("date_heure", mode="before")
    @classmethod
    def _parse_datetime(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v))

    @field_validator("consommation", "prevision_j1", "prevision_j", mode="before")
    @classmethod
    def _coerce_optional_int(cls, v: Any) -> Any:
        # pandas turns None into NaN when records go through a DataFrame
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if isinstance(v, str):
            return int(float(v))
        return int(v)


# Continuous variables: spatially averaged (mean) per hour.
WEATHER_VARS_MEAN = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "is_day",
)
# Categorical variable: majority vote (mode) per hour.
WEATHER_VARS_MODE = ("weather_code",)
WEATHER_UNITS = {
    "temperature_2m": "c",
    "apparent_temperature": "c",
    "relative_humidity_2m": "pct",
    "pressure_msl": "hpa",
    "precipitation": "mm",
    "wind_speed_10m": "ms",
    "shortwave_radiation": "wm2",
    "cloud_cover": "pct",
    "is_day": "fraction",
    "weather_code": "code",
}
WEATHER_FLOAT_VARS = WEATHER_VARS_MEAN
WEATHER_INT_VARS = WEATHER_VARS_MODE + ("is_day",)


class WeatherRecord(BaseModel):
    """One archived forecast grid point (hourly weather vars), strictly typed."""

    model_config = ConfigDict(extra="forbid")

    time: datetime
    latitude: float
    longitude: float
    temperature_2m: float | None = None
    apparent_temperature: float | None = None
    relative_humidity_2m: float | None = None
    pressure_msl: float | None = None
    precipitation: float | None = None
    wind_speed_10m: float | None = None
    shortwave_radiation: float | None = None
    cloud_cover: float | None = None
    is_day: int | None = None
    weather_code: int | None = None
    model: str

    @field_validator("time", mode="before")
    @classmethod
    def _parse_time(cls, v: Any) -> datetime:
        dt = v if isinstance(v, datetime) else datetime.fromisoformat(str(v))
        # Open-Meteo with timezone=UTC returns naive ISO strings: pin them to UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt

    @field_validator(*WEATHER_FLOAT_VARS, mode="before")
    @classmethod
    def _coerce_optional_float(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return float(v)

    @field_validator(*WEATHER_INT_VARS, mode="before")
    @classmethod
    def _coerce_optional_int(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if isinstance(v, str):
            return int(float(v))
        return int(v)


@dataclass
class ValidationReport:
    input_rows: int = 0
    valid_rows: int = 0
    rejected_rows: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.valid_rows}/{self.input_rows} rows valid, "
            f"{self.rejected_rows} rejected"
        )


def _validate_with(
    model_cls: type[BaseModel], raw: pd.DataFrame
) -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport(input_rows=len(raw))
    clean: list[dict[str, Any]] = []
    for i, record in enumerate(raw.to_dict("records")):
        try:
            clean.append(model_cls.model_validate(record).model_dump())
        except ValidationError as exc:
            report.rejected_rows += 1
            if len(report.errors) < 20:
                report.errors.append(f"row {i}: {exc.errors()[0]['msg']}")
    report.valid_rows = len(clean)
    logger.info("validation: %s", report.summary())
    if report.errors:
        logger.warning("first validation errors: %s", report.errors[:5])
    return pd.DataFrame(clean), report


def validate_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate raw eco2mix records row by row against the strict schema."""
    return _validate_with(Eco2mixRecord, raw)


def validate_weather(raw: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate raw archived-forecast records row by row."""
    return _validate_with(WeatherRecord, raw)


def aggregate_hourly(
    clean: pd.DataFrame,
    *,
    perimetre: str = DEFAULT_PERIMETRE,
    natures: tuple[str, ...] = DEFAULT_NATURES,
) -> pd.DataFrame:
    """Filter to a perimeter/natures and aggregate the 30-min series to hourly.

    UTC hourly means of the MW values; NaN where the source had nulls
    (consommation is only published on :00/:30, forecasts on every quarter).
    """
    mask = (clean["perimetre"] == perimetre) & clean["nature"].isin(natures)
    sub = clean.loc[mask].copy()
    sub["date_heure"] = pd.to_datetime(sub["date_heure"], utc=True)
    sub = sub.set_index("date_heure").sort_index()
    hourly = sub.resample("1h")[["consommation", "prevision_j1", "prevision_j"]].mean()
    hourly.index.name = "date_heure_utc"
    return hourly


def _mode_or_none(s: pd.Series) -> Any:
    m = s.dropna().mode()
    return int(m.iloc[0]) if len(m) else None


def aggregate_weather_national(clean: pd.DataFrame) -> pd.DataFrame:
    """National per-UTC-hour aggregation over the grid.

    Continuous variables: simple mean. ``weather_code``: majority (mode).
    """
    sub = clean.copy()
    sub["time"] = pd.to_datetime(sub["time"], utc=True)
    mean_vars = [c for c in WEATHER_VARS_MEAN if c in sub.columns]
    hourly = sub.groupby("time")[mean_vars].mean()
    hourly.columns = [f"{c}_{WEATHER_UNITS[c]}" for c in mean_vars]
    for var in WEATHER_VARS_MODE:
        if var in sub.columns:
            hourly[f"{var}_{WEATHER_UNITS[var]}"] = sub.groupby("time")[var].agg(
                _mode_or_none
            )
    hourly.index.name = "date_heure_utc"
    return hourly.sort_index()


def build_hourly_dataset(
    raw: pd.DataFrame,
    *,
    perimetre: str = DEFAULT_PERIMETRE,
    natures: tuple[str, ...] = DEFAULT_NATURES,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Full pipeline: validate raw eco2mix records, then aggregate to hourly UTC."""
    clean, report = validate_raw(raw)
    return aggregate_hourly(clean, perimetre=perimetre, natures=natures), report


def build_weather_dataset(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Full pipeline: validate raw forecast records, then national hourly means."""
    clean, report = validate_weather(raw)
    return aggregate_weather_national(clean), report
