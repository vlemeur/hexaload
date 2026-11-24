"""Shared fixtures: small in-memory raw eco2mix-like records (offline)."""

from pathlib import Path

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
