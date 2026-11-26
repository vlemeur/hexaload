"""Monitoring brick: daily scoring of published forecasts + drift detection.

Pure logic (no network): everything reads the on-disk cache. The scheduled
run happens in the forecast GitHub Action, which refreshes the cache first.
"""

from .cusum import CusumState, cusum_reset, cusum_update
from .score_live import daily_scores, json_records
from .status import compute_status

__all__ = [
    "CusumState",
    "cusum_reset",
    "cusum_update",
    "compute_status",
    "daily_scores",
    "json_records",
]
