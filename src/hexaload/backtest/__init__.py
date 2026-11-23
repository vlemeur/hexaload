"""Walk-forward backtesting: sktime splits + naive baselines."""

from hexaload.backtest.baselines import (
    predict_naive,
    predict_rte_prevision_j1,
    predict_seasonal_naive,
)
from hexaload.backtest.walkforward import Split, walk_forward_splits

__all__ = [
    "Split",
    "predict_naive",
    "predict_rte_prevision_j1",
    "predict_seasonal_naive",
    "walk_forward_splits",
]
