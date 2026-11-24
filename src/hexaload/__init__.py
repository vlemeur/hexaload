"""hexaload: an honest benchmark of national French electricity load forecasting."""

import os

# libomp (xgboost) and torch use separate native threadpools; loaded together
# they segfault at interpreter shutdown on macOS. Pin OpenMP to 1 thread —
# xgboost stays fast enough for our fold sizes, and torch has its own pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
