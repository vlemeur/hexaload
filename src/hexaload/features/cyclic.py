"""Cyclic (sin/cos) encodings of calendar time — preserves continuity.

Computed in Europe/Paris local time, so the one-hour DST shift of the evening
peak is handled natively by the hour encoding (and the 23h -> 0h wrap-around
stays continuous, unlike raw one-hot hour). Day-of-week and day-of-year are
encoded the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlock, as_utc_hourly_index
from .calendar import TZ_PARIS

CYC_COLUMNS = (
    "cyc_heure_sin",
    "cyc_heure_cos",
    "cyc_jour_semaine_sin",
    "cyc_jour_semaine_cos",
    "cyc_jour_annee_sin",
    "cyc_jour_annee_cos",
)


def cyclic_block(idx: pd.DatetimeIndex) -> FeatureBlock:
    """Cyclic encodings on a UTC hourly index (delays 0: deterministic)."""
    idx = as_utc_hourly_index(idx)
    local = idx.tz_convert(TZ_PARIS)
    hour = local.hour + local.minute / 60.0 + local.second / 3600.0
    two_pi = 2.0 * np.pi
    frame = pd.DataFrame(
        {
            "cyc_heure_sin": np.sin(two_pi * hour / 24.0),
            "cyc_heure_cos": np.cos(two_pi * hour / 24.0),
            "cyc_jour_semaine_sin": np.sin(two_pi * local.day_of_week / 7.0),
            "cyc_jour_semaine_cos": np.cos(two_pi * local.day_of_week / 7.0),
            "cyc_jour_annee_sin": np.sin(two_pi * (local.day_of_year - 1) / 365.25),
            "cyc_jour_annee_cos": np.cos(two_pi * (local.day_of_year - 1) / 365.25),
        },
        index=idx,
    )
    return FeatureBlock("cyclic", frame, {c: 0 for c in CYC_COLUMNS})
