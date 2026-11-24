"""Feature engineering for hexaload — leakage-safe feature blocks.

Every feature column declares an availability delay (hours): it may only use
information known at ``t - delay``. ``build_feature_table(..., as_of=origin)``
masks each column accordingly, so a walk-forward fold can never see the
future through the feature table.
"""

from .availability import Manifest, apply_as_of, assert_horizon_safe, stack_blocks
from .base import FeatureBlock, as_utc_hourly_index
from .calendar import CAL_COLUMNS, calendar_block
from .cyclic import CYC_COLUMNS, cyclic_block
from .lags import lags_block
from .table import DEFAULT_BLOCKS, DEFAULT_HORIZON, build_feature_table
from .weather import (
    CDD_BASE_C,
    HDD_BASE_C,
    load_grid_weights,
    national_weather_frame,
    weather_block,
)

__all__ = [
    "CAL_COLUMNS",
    "CDD_BASE_C",
    "CYC_COLUMNS",
    "DEFAULT_BLOCKS",
    "DEFAULT_HORIZON",
    "HDD_BASE_C",
    "FeatureBlock",
    "Manifest",
    "apply_as_of",
    "as_utc_hourly_index",
    "assert_horizon_safe",
    "build_feature_table",
    "calendar_block",
    "cyclic_block",
    "lags_block",
    "load_grid_weights",
    "national_weather_frame",
    "stack_blocks",
    "weather_block",
]
