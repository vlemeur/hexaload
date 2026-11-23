"""Anti-leakage tests for the walk-forward splits (offline)."""

import numpy as np
import pandas as pd
import pytest

from hexaload.backtest.walkforward import walk_forward_splits


def hourly(n: int, start: str = "2023-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def test_splits_are_leakage_free_and_ordered() -> None:
    y = hourly(24 * 400)
    splits = walk_forward_splits(
        y,
        start="2023-01-01",
        end="2023-12-01",
        horizon=24,
        window_length=24 * 100,
        step_length=24 * 10,
    )
    assert len(splits) > 1
    starts = [s.test_start for s in splits]
    assert starts == sorted(starts)
    for s in splits:
        assert len(s.test) == 24
        assert (s.train < s.test.min()).all()  # no future in train
        # test block = 24 consecutive hourly steps
        assert (s.test[1:] - s.test[:-1] == pd.Timedelta(hours=1)).all()
        assert len(s.train) == 24 * 100
        assert (s.test >= pd.Timestamp("2023-01-01", tz="UTC")).all()
        assert (s.test < pd.Timestamp("2023-12-02", tz="UTC")).all()


def test_expanding_window_grows() -> None:
    y = hourly(24 * 400)
    splits = walk_forward_splits(
        y,
        start="2023-01-01",
        end="2023-12-01",
        horizon=24,
        window_length=24 * 50,
        step_length=24 * 20,
        expanding=True,
    )
    lengths = [len(s.train) for s in splits]
    assert lengths == sorted(lengths)
    assert lengths[0] == 24 * 50
    assert lengths[-1] > lengths[0]
    for s in splits:
        assert (s.train < s.test.min()).all()


def test_not_enough_data_raises() -> None:
    y = hourly(200)
    with pytest.raises(ValueError, match="not enough data"):
        walk_forward_splits(
            y,
            start="2023-01-01",
            end="2023-01-09",
            horizon=24,
            window_length=192,
            step_length=24,
        )


def test_invalid_horizon_raises() -> None:
    y = hourly(24 * 30)
    with pytest.raises(ValueError, match="horizon"):
        walk_forward_splits(
            y,
            start="2023-01-01",
            end="2023-01-31",
            horizon=0,
            window_length=24,
            step_length=24,
        )
