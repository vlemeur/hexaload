"""Sanity tests for the metrics wrapper (offline)."""

import numpy as np
import pandas as pd
import pytest

from hexaload.metrics import crps, mae, mase, rmse


def series(vals: list[float], start: str = "2023-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(vals), freq="1h", tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


def test_mae_rmse_known_values() -> None:
    y = series([1, 2, 3, 4])
    p = series([1, 2, 3, 6])
    assert mae(y, p) == pytest.approx(0.5)
    assert rmse(y, p) == pytest.approx(1.0)


def test_alignment_drops_nan_pairs() -> None:
    y = series([1.0, np.nan, 3.0])
    p = series([1.0, 2.0, 3.0])
    assert mae(y, p) == pytest.approx(0.0)


def test_empty_overlap_raises() -> None:
    y = series([1.0, np.nan])
    p = series([np.nan, 2.0])
    with pytest.raises(ValueError, match="no overlapping"):
        mae(y, p)


def test_mase_perfect_forecast_is_zero() -> None:
    y_train = series(list(50 + 10 * np.sin(np.arange(96) / 24 * 2 * np.pi)))
    y = series([10.0, 20.0, 30.0], start="2023-01-05")  # after y_train
    assert mase(y, y.copy(), y_train, sp=24) == pytest.approx(0.0)


def test_mase_matches_sktime_directly() -> None:
    # The wrapper must be a thin pass-through of sktime's MASE, not a recode.
    from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

    rng = np.random.default_rng(0)
    y_train = series(list(50 + rng.normal(0, 5, 96)))
    y = series(list(50 + rng.normal(0, 5, 24)), start="2023-01-05")
    pred = y + rng.normal(0, 3, 24)
    expected = float(
        MeanAbsoluteScaledError(sp=24)(
            y.to_numpy(), pred.to_numpy(), y_train=y_train.to_numpy()
        )
    )
    assert mase(y, pred, y_train, sp=24) == pytest.approx(expected)


def test_mase_tolerates_nan_in_train() -> None:
    # Regression: sktime's scale must stay finite even if y_train has holes.
    rng = np.random.default_rng(1)
    vals = [float(v) for v in 50 + rng.normal(0, 5, 96)]
    vals[10] = np.nan
    y_train = series(vals)
    y = series(list(50 + rng.normal(0, 5, 24)), start="2023-01-05")
    pred = y + rng.normal(0, 3, 24)
    value = mase(y, pred, y_train, sp=24)
    assert np.isfinite(value)
    scale = y_train.dropna().diff(24).abs().mean()
    assert value == pytest.approx((y - pred).abs().mean() / scale, rel=1e-6)


def test_crps_single_member_equals_mae() -> None:
    y = series([1.0, 2.0, 3.0])
    samples = pd.DataFrame({"m0": [2.0, 2.0, 5.0]}, index=y.index)
    assert crps(y, samples) == pytest.approx(mae(y, samples["m0"]))


def test_crps_perfect_ensemble_is_zero() -> None:
    y = series([1.0, 2.0, 3.0])
    samples = pd.DataFrame({f"m{i}": y for i in range(5)})
    assert crps(y, samples) == pytest.approx(0.0, abs=1e-12)


def test_crps_penalizes_bias_and_is_nonnegative() -> None:
    y = series([0.0, 0.0, 0.0])
    good = pd.DataFrame(
        {f"m{i}": [v] for i, v in enumerate([-1.0, 0.0, 1.0])}, index=y.index
    )
    biased = good + 5.0
    assert crps(y, good) >= 0.0
    assert crps(y, biased) > crps(y, good)
