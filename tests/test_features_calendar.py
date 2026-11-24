"""Calendar features: holidays, bridge days, school vacations, DST (offline).

School vacations use a small synthetic CSV in the official data.gouv.fr
schema (date, vacances_zone_a/b/c, nom_vacances). A separate test checks a
few official 2025-2026 dates against the embedded CSV when it exists.
"""

from pathlib import Path

import pandas as pd
import pytest

from hexaload.features.calendar import (
    CAL_COLUMNS,
    SCHOOL_VACATIONS_CSV,
    calendar_block,
    validate_school_vacations,
)


def hourly(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1h", tz="UTC")


def block_on(idx: pd.DatetimeIndex, school_csv: Path):
    return calendar_block(idx, school_csv=school_csv).frame


def day_val(frame: pd.DataFrame, day: str, col: str) -> int:
    ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=12)
    return int(frame.loc[ts, col])


def test_public_holidays_and_neighbours(school_csv: Path) -> None:
    idx = hourly("2024-05-06", "2024-05-12 23:00")
    f = block_on(idx, school_csv)
    assert day_val(f, "2024-05-08", "cal_ferie") == 1  # Wed, Victory in Europe Day
    assert day_val(f, "2024-05-09", "cal_ferie") == 1  # Thu, Ascension
    assert day_val(f, "2024-05-10", "cal_ferie") == 0
    assert day_val(f, "2024-05-07", "cal_lendemain_ferie") == 1
    assert day_val(f, "2024-05-10", "cal_veille_ferie") == 1
    assert day_val(f, "2024-05-06", "cal_ferie") == 0


def test_bridge_day_definition(school_csv: Path) -> None:
    # Friday 2024-05-10 sits between Ascension (Thu) and the weekend -> pont
    idx = hourly("2024-05-08", "2024-05-20 23:00")
    f = block_on(idx, school_csv)
    assert day_val(f, "2024-05-10", "cal_pont") == 1
    assert day_val(f, "2024-05-17", "cal_pont") == 0  # ordinary Friday
    assert day_val(f, "2024-05-09", "cal_pont") == 0  # holiday itself, not a pont
    # Friday 2025-05-02 after Labour Day (Thu 2025-05-01) -> pont
    idx2 = hourly("2025-04-30", "2025-05-05 23:00")
    f2 = block_on(idx2, school_csv)
    assert day_val(f2, "2025-05-02", "cal_pont") == 1


def test_school_vacations_zones(school_csv: Path) -> None:
    idx = hourly("2025-10-15", "2026-03-10 23:00")
    f = block_on(idx, school_csv)
    assert day_val(f, "2025-10-20", "cal_vacances_zone_a") == 1  # Toussaint, all
    assert day_val(f, "2025-10-20", "cal_vacances_zone_b") == 1
    assert day_val(f, "2025-10-20", "cal_vacances_zone_c") == 1
    assert day_val(f, "2025-10-17", "cal_vacances_zone_a") == 0  # before
    # winter 2026: A 07-22 Feb, B 14 Feb-01 Mar, C 21 Feb-08 Mar
    assert day_val(f, "2026-02-10", "cal_vacances_zone_a") == 1
    assert day_val(f, "2026-02-10", "cal_vacances_zone_b") == 0
    assert day_val(f, "2026-02-16", "cal_vacances_zone_a") == 1
    assert day_val(f, "2026-02-16", "cal_vacances_zone_b") == 1
    assert day_val(f, "2026-02-16", "cal_vacances_zone_c") == 0
    assert day_val(f, "2026-02-25", "cal_vacances_zone_c") == 1
    assert day_val(f, "2026-02-25", "cal_vacances_zone_a") == 0


def test_dst_flag(school_csv: Path) -> None:
    # 2024-03-31: switch to summer time at 01:00 UTC (02:00 CET -> 03:00 CEST)
    idx = hourly("2024-03-31 00:00", "2024-03-31 03:00")
    f = block_on(idx, school_csv)
    assert int(f.loc["2024-03-31 00:00", "cal_heure_ete"]) == 0
    assert int(f.loc["2024-03-31 01:00", "cal_heure_ete"]) == 1
    # 2024-10-27: switch back at 01:00 UTC (03:00 CEST -> 02:00 CET)
    idx2 = hourly("2024-10-27 00:00", "2024-10-27 03:00")
    f2 = block_on(idx2, school_csv)
    assert int(f2.loc["2024-10-27 00:00", "cal_heure_ete"]) == 1
    assert int(f2.loc["2024-10-27 01:00", "cal_heure_ete"]) == 0


def test_truncated_index_gives_same_flags(school_csv: Path) -> None:
    """Border flags must not depend on the caller's index range."""
    full = block_on(hourly("2024-05-06", "2024-05-12 23:00"), school_csv)
    part = block_on(hourly("2024-05-08", "2024-05-10 23:00"), school_csv)
    common = part.index
    pd.testing.assert_frame_equal(full.loc[common, list(CAL_COLUMNS)], part)


def test_validate_school_vacations_strict() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2025-10-20",
                "vacances_zone_a": "true",
                "vacances_zone_b": False,
                "vacances_zone_c": 0,
                "nom_vacances": "Vacances de la Toussaint",
            },
            # official CSV: empty nom_vacances outside vacation days (NaN)
            {
                "date": "2025-10-21",
                "vacances_zone_a": False,
                "vacances_zone_b": False,
                "vacances_zone_c": False,
                "nom_vacances": None,
            },
            {
                "date": "not-a-date",
                "vacances_zone_a": True,
                "vacances_zone_b": True,
                "vacances_zone_c": True,
                "nom_vacances": "x",
            },
        ]
    )
    clean = validate_school_vacations(raw)
    assert len(clean) == 2  # only the bad date is rejected
    assert clean.index[0] == pd.Timestamp("2025-10-20")
    assert bool(clean["vacances_zone_a"].iloc[0]) is True
    assert clean["nom_vacances"].iloc[1] == ""  # NaN coerced to empty string


@pytest.mark.skipif(
    not SCHOOL_VACATIONS_CSV.exists(), reason="embedded CSV not built yet"
)
def test_embedded_csv_matches_official_2025_2026() -> None:
    """Official dates (education.gouv.fr, 2025-2026): Toussaint 18 Oct-02 Nov
    2025 for all zones; winter 2026: A 07-22 Feb, B 14 Feb-01 Mar, C 21 Feb-08 Mar."""
    f = calendar_block(hourly("2025-10-15", "2026-03-10 23:00")).frame
    assert day_val(f, "2025-10-25", "cal_vacances_zone_a") == 1
    assert day_val(f, "2025-10-25", "cal_vacances_zone_b") == 1
    assert day_val(f, "2025-10-25", "cal_vacances_zone_c") == 1
    assert day_val(f, "2025-10-17", "cal_vacances_zone_a") == 0
    assert day_val(f, "2026-02-10", "cal_vacances_zone_a") == 1
    assert day_val(f, "2026-02-10", "cal_vacances_zone_b") == 0
    assert day_val(f, "2026-02-16", "cal_vacances_zone_b") == 1
    assert day_val(f, "2026-02-16", "cal_vacances_zone_c") == 0
    assert day_val(f, "2026-02-25", "cal_vacances_zone_c") == 1
