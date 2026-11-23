"""Shared fixtures: small in-memory raw eco2mix-like records (offline)."""

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
