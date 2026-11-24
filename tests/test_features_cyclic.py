"""Cyclic encodings: continuity at wrap-arounds, local-time DST (offline)."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from hexaload.features.cyclic import CYC_COLUMNS, cyclic_block


def hourly(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1h", tz="UTC")


def ts(s: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(s, tz="UTC"))


def hour_of(t: pd.Timestamp, frame: pd.DataFrame) -> float:
    s = frame.loc[t, "cyc_heure_sin"]
    c = frame.loc[t, "cyc_heure_cos"]
    return float((np.arctan2(s, c) * 24.0 / (2.0 * np.pi)) % 24.0)


def dist(frame: pd.DataFrame, t1: str, t2: str, prefix: str) -> float:
    a = frame.loc[t1, [f"{prefix}_sin", f"{prefix}_cos"]].to_numpy(float)
    b = frame.loc[t2, [f"{prefix}_sin", f"{prefix}_cos"]].to_numpy(float)
    return float(np.linalg.norm(a - b))


def test_hour_wraparound_is_continuous() -> None:
    # summer: local = UTC+2; 2024-06-03 21:00 UTC = 23:00 local, 22:00 UTC = 00:00
    f = cyclic_block(hourly("2024-06-03 08:00", "2024-06-04 01:00")).frame
    d_2324_0 = dist(f, "2024-06-03 21:00", "2024-06-03 22:00", "cyc_heure")
    d_2324_12 = dist(f, "2024-06-03 21:00", "2024-06-03 10:00", "cyc_heure")
    assert d_2324_0 < 0.3 < d_2324_12


def test_hour_encoding_is_local_and_dst_aware() -> None:
    # 2024-03-31 01:00 UTC = 03:00 local (02:00 does not exist, spring forward)
    f = cyclic_block(hourly("2024-03-31 00:00", "2024-03-31 02:00")).frame
    assert hour_of(ts("2024-03-31 00:00"), f) == pytest.approx(1.0)
    assert hour_of(ts("2024-03-31 01:00"), f) == pytest.approx(3.0)
    # 2024-10-27 01:00 UTC = 02:00 local (clocks go back)
    f2 = cyclic_block(hourly("2024-10-27 00:00", "2024-10-27 02:00")).frame
    assert hour_of(ts("2024-10-27 00:00"), f2) == pytest.approx(2.0)
    assert hour_of(ts("2024-10-27 01:00"), f2) == pytest.approx(2.0)


def test_day_of_year_wraparound_is_continuous() -> None:
    # local days: 2024-12-30 23:00 UTC = 2024-12-31 00:00
    # local; 2024-12-31 23:00 UTC = 2025-01-01 00:00 local
    f = cyclic_block(hourly("2024-07-01 00:00", "2025-01-02 00:00")).frame
    d_wrap = dist(f, "2024-12-30 23:00", "2024-12-31 23:00", "cyc_jour_annee")
    d_far = dist(f, "2024-12-31 23:00", "2024-07-02 12:00", "cyc_jour_annee")
    assert d_wrap < 0.1 < d_far


def test_columns_and_delays() -> None:
    b = cyclic_block(hourly("2024-01-01", "2024-01-02 23:00"))
    assert tuple(b.columns) == CYC_COLUMNS
    assert set(b.delays.values()) == {0}
