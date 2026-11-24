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
from .table import DEFAULT_BLOCKS, build_feature_table

__all__ = [
    "CAL_COLUMNS",
    "CYC_COLUMNS",
    "DEFAULT_BLOCKS",
    "FeatureBlock",
    "Manifest",
    "apply_as_of",
    "as_utc_hourly_index",
    "assert_horizon_safe",
    "build_feature_table",
    "calendar_block",
    "cyclic_block",
    "stack_blocks",
]
