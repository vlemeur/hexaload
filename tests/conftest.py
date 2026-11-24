"""Shared fixtures: small in-memory raw eco2mix-like records (offline)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_records() -> list[dict]:
    return [
        {
            "date_heure": "2023-01-01T00:15:00+00:00",
            "date": "2023-01-01",
            "heure": "01:15",
            "perimetre": "France",
            "nature": "Données définitives",
            "consommation": None,
            "prevision_j1": 49550,
            "prevision_j": 49900,
            "gaz_cogen": "846",  # stringly-typed extra column, kept as-is
        },
        {
            "date_heure": "2023-01-01T00:45:00+00:00",
            "date": "2023-01-01",
            "heure": "01:45",
            "perimetre": "France",
            "nature": "Données définitives",
            "consommation": 50196,
            "prevision_j1": "48300",  # stringly-typed core column
            "prevision_j": 48900,
        },
        {
            "date_heure": "2023-01-01T01:15:00+00:00",
            "date": "2023-01-01",
            "heure": "02:15",
            "perimetre": "Corse",
            "nature": "Données définitives",
            "consommation": 500,
            "prevision_j1": 480,
            "prevision_j": 490,
        },
    ]


@pytest.fixture
def raw_df(raw_records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(raw_records)


@pytest.fixture
def school_csv(tmp_path: Path) -> Path:
    """Synthetic school vacations in the official data.gouv.fr schema
    (date, vacances_zone_a/b/c, nom_vacances): Toussaint 2025 all zones,
    winter 2026 zone A 07-22 Feb, B 14 Feb-01 Mar, C 21 Feb-08 Mar."""
    rows = []
    for d in pd.date_range("2025-01-01", "2026-12-31", freq="1D"):
        a = b = c = False
        nom = "Hors vacances"
        if pd.Timestamp("2025-10-18") <= d <= pd.Timestamp("2025-11-02"):
            a = b = c = True
            nom = "Vacances de la Toussaint"
        if pd.Timestamp("2026-02-07") <= d <= pd.Timestamp("2026-02-22"):
            a = True
            nom = "Vacances d'hiver"
        if pd.Timestamp("2026-02-14") <= d <= pd.Timestamp("2026-03-01"):
            b = True
            nom = "Vacances d'hiver"
        if pd.Timestamp("2026-02-21") <= d <= pd.Timestamp("2026-03-08"):
            c = True
            nom = "Vacances d'hiver"
        rows.append(
            {
                "date": d.date().isoformat(),
                "vacances_zone_a": a,
                "vacances_zone_b": b,
                "vacances_zone_c": c,
                "nom_vacances": nom,
            }
        )
    path = tmp_path / "school_vacations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def hourly_idx() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=24 * 40, freq="1h", tz="UTC")


@pytest.fixture
def y_load(hourly_idx: pd.DatetimeIndex) -> pd.Series:
    """Synthetic load with a daily cycle + weekly modulation."""
    rng = np.random.default_rng(0)
    h = hourly_idx.hour
    dow = hourly_idx.day_of_week
    y = (
        50_000
        + 5_000 * np.sin(2 * np.pi * (h - 9) / 24)
        - 3_000 * (dow >= 5)
        + rng.normal(0, 200, len(hourly_idx))
    )
    return pd.Series(y, index=hourly_idx, name="consommation")


@pytest.fixture
def wx_national(hourly_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic national weather frame (unit-suffixed columns)."""
    n = len(hourly_idx)
    t = (
        10.0
        + 8.0 * np.sin(2 * np.pi * (hourly_idx.hour - 14) / 24)
        - 5.0 * np.sin(2 * np.pi * hourly_idx.day_of_year / 365)
    )
    return pd.DataFrame(
        {
            "temperature_2m_c": t,
            "apparent_temperature_c": t - 1.0,
            "relative_humidity_2m_pct": np.full(n, 60.0),
            "pressure_msl_hpa": 1013.0 + np.sin(np.arange(n) / 8.0),
            "precipitation_mm": np.zeros(n),
            "wind_speed_10m_ms": np.full(n, 3.0),
            "shortwave_radiation_wm2": np.where(
                (hourly_idx.hour >= 7) & (hourly_idx.hour <= 19), 400.0, 0.0
            ),
            "cloud_cover_pct": np.full(n, 30.0),
            "is_day_fraction": np.where(
                (hourly_idx.hour >= 7) & (hourly_idx.hour <= 19), 1.0, 0.0
            ),
            "weather_code_code": np.full(n, 1),
        },
        index=hourly_idx,
    )
