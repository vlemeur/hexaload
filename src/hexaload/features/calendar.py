"""French calendar features — data, not code.

Public holidays come from the ``holidays`` package; school vacations by zone
(A/B/C) from the official data.gouv.fr dataset "Vacances scolaires par zones"
(github.com/AntoineAugusti/vacances-scolaires), embedded as a versioned CSV
under ``features/data/`` by ``scripts/build_school_calendar.py`` so the
backtest runs fully offline.

All day-level flags are computed in Europe/Paris local time (a public holiday
is a local day) and mapped back onto the UTC hourly index. The daily grid is
built with a 2-day margin on each side so border flags (veille/lendemain,
pont) are identical whether the caller passes a full or a truncated index —
this is what makes the truncation no-leakage test pass by construction.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import holidays as holidays_lib
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .base import FeatureBlock, as_utc_hourly_index

logger = logging.getLogger(__name__)

TZ_PARIS = "Europe/Paris"
DATA_DIR = Path(__file__).parent / "data"
SCHOOL_VACATIONS_CSV = DATA_DIR / "school_vacations_fr.csv"

CAL_COLUMNS = (
    "cal_ferie",
    "cal_veille_ferie",
    "cal_lendemain_ferie",
    "cal_pont",
    "cal_vacances_zone_a",
    "cal_vacances_zone_b",
    "cal_vacances_zone_c",
    "cal_heure_ete",
)


class SchoolVacationDay(BaseModel):
    """One row of the official school-vacations CSV, strictly typed."""

    model_config = ConfigDict(extra="forbid")

    date: date
    vacances_zone_a: bool
    vacances_zone_b: bool
    vacances_zone_c: bool
    # empty outside vacation days in the official CSV (NaN once read back)
    nom_vacances: str = ""

    @field_validator("date", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v))

    @field_validator("nom_vacances", mode="before")
    @classmethod
    def _empty_to_default(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return str(v)


def validate_school_vacations(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate raw school-vacation records row by row (strict pydantic).

    Network-free: feed it the raw CSV as a DataFrame. Returns a clean frame
    indexed by naive local ``date`` with boolean zone flags.
    """
    clean: list[dict[str, Any]] = []
    rejected = 0
    for i, record in enumerate(raw.to_dict("records")):
        try:
            clean.append(SchoolVacationDay.model_validate(record).model_dump())
        except ValidationError as exc:
            rejected += 1
            if rejected <= 5:
                logger.warning(
                    "school-vacations row %d rejected: %s", i, exc.errors()[0]["msg"]
                )
    if rejected:
        logger.warning("school-vacations validation: %d rows rejected", rejected)
    df = pd.DataFrame(clean)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_school_vacations(path: Path = SCHOOL_VACATIONS_CSV) -> pd.DataFrame:
    """Load the embedded school-vacations CSV (validated). Offline."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — embed the official calendar with: "
            "uv run python scripts/build_school_calendar.py"
        )
    return validate_school_vacations(pd.read_csv(path))


def _daily_flags(days: pd.DatetimeIndex, school_csv: Path) -> pd.DataFrame:
    """Day-level flags on a continuous Europe/Paris-midnight index.

    Bridge day ("pont") definition: a working day (Mon-Fri, not a public
    holiday) wedged between a public holiday and a rest day (Sat/Sun) — i.e.
    previous day is a holiday and next day is Sat/Sun (classic Thursday
    holiday -> Friday bridge), or the mirror image, or between two holidays.
    """
    years = list(range(days.year.min(), days.year.max() + 1))
    fr = holidays_lib.country_holidays("FR", years=years)
    is_holiday = pd.Series([d.date() in fr for d in days], index=days, dtype="int8")
    workday = (days.day_of_week < 5) & (is_holiday == 0)
    prev_h = is_holiday.shift(1, fill_value=0)
    next_h = is_holiday.shift(-1, fill_value=0)
    prev_is_weekend = (days - pd.Timedelta(days=1)).day_of_week >= 5
    next_is_weekend = (days + pd.Timedelta(days=1)).day_of_week >= 5
    is_bridge = (
        workday
        & (
            (prev_h.eq(1) & next_is_weekend)
            | (next_h.eq(1) & prev_is_weekend)
            | (prev_h.eq(1) & next_h.eq(1))
        )
    ).astype("int8")

    vac = load_school_vacations(school_csv)
    days_naive = days.tz_localize(None)
    zones = {}
    for zone, col in (
        ("a", "vacances_zone_a"),
        ("b", "vacances_zone_b"),
        ("c", "vacances_zone_c"),
    ):
        zones[zone] = (
            vac[col].reindex(pd.DatetimeIndex(days_naive)).fillna(False).astype("int8")
        )
    return pd.DataFrame(
        {
            "cal_ferie": is_holiday,
            "cal_veille_ferie": prev_h.astype("int8"),
            "cal_lendemain_ferie": next_h.astype("int8"),
            "cal_pont": is_bridge,
            "cal_vacances_zone_a": zones["a"].to_numpy(),
            "cal_vacances_zone_b": zones["b"].to_numpy(),
            "cal_vacances_zone_c": zones["c"].to_numpy(),
        },
        index=days,
    )


def calendar_block(
    idx: pd.DatetimeIndex, *, school_csv: Path = SCHOOL_VACATIONS_CSV
) -> FeatureBlock:
    """Calendar features on a UTC hourly index (all delays 0: deterministic)."""
    idx = as_utc_hourly_index(idx)
    local = idx.tz_convert(TZ_PARIS)
    day_of = local.normalize()
    # 2-day margin so border flags do not depend on the caller's index range
    days = pd.date_range(
        day_of.min() - pd.Timedelta(days=2),
        day_of.max() + pd.Timedelta(days=2),
        freq="1D",
        tz=TZ_PARIS,
    )
    daily = _daily_flags(days, school_csv=school_csv)
    frame = daily.reindex(day_of)
    frame.index = idx
    # UTC offset of local time, vectorized: +2h <=> summer time (CEST)
    offset_h = (
        local.tz_localize(None) - idx.tz_localize(None)
    ).total_seconds() / 3600.0
    frame["cal_heure_ete"] = (offset_h == 2.0).astype("int8")
    frame = frame[list(CAL_COLUMNS)]
    return FeatureBlock("calendar", frame, {c: 0 for c in CAL_COLUMNS})
