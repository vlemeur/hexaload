"""Tests for hexaload.data.validate — pure, offline."""

import pandas as pd
import pytest

from hexaload.data.validate import (
    Eco2mixRecord,
    aggregate_hourly,
    build_hourly_dataset,
    validate_raw,
)


def test_record_coerces_stringly_typed_ints():
    rec = Eco2mixRecord.model_validate(
        {
            "date_heure": "2023-01-01T00:45:00+00:00",
            "date": "2023-01-01",
            "heure": "01:45",
            "perimetre": "France",
            "nature": "Données définitives",
            "consommation": 50196,
            "prevision_j1": "48300",
            "prevision_j": None,
        }
    )
    assert rec.prevision_j1 == 48300
    assert rec.prevision_j is None
    assert rec.date_heure.tzinfo is not None


def test_validate_raw_reports_rejections(raw_df: pd.DataFrame):
    bad = raw_df.copy()
    bad.loc[0, "date_heure"] = "not-a-date"
    clean, report = validate_raw(bad)
    assert report.input_rows == 3
    assert report.valid_rows == 2
    assert report.rejected_rows == 1
    assert len(clean) == 2


def test_aggregate_hourly_filters_and_averages(raw_df: pd.DataFrame):
    clean, _ = validate_raw(raw_df)
    hourly = aggregate_hourly(clean)
    # only the two France/définitives rows, both in the 2023-01-01 00:00 UTC hour
    assert len(hourly) == 1
    row = hourly.iloc[0]
    assert row["consommation"] == pytest.approx(50196.0)  # None skipped by mean
    assert row["prevision_j1"] == pytest.approx((49550 + 48300) / 2)


def test_build_hourly_dataset_end_to_end(raw_df: pd.DataFrame):
    hourly, report = build_hourly_dataset(raw_df)
    assert report.valid_rows == 3
    assert hourly.index.name == "date_heure_utc"
    assert list(hourly.columns) == ["consommation", "prevision_j1", "prevision_j"]
