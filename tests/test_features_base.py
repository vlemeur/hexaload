"""FeatureBlock contract + the truncation no-leakage test (offline)."""

import numpy as np
import pandas as pd
import pytest

from hexaload.features.base import FeatureBlock, as_utc_hourly_index


def hourly_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def toy_block(idx: pd.DatetimeIndex, delays: dict[str, int]) -> FeatureBlock:
    frame = pd.DataFrame(
        {c: np.arange(len(idx), dtype=float) for c in delays}, index=idx
    )
    return FeatureBlock("toy", frame, delays, reads_target=True)


def test_index_must_be_tz_aware_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        as_utc_hourly_index(pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3)))
    idx = pd.date_range("2024-01-01", periods=3, tz="Europe/Paris")
    assert str(as_utc_hourly_index(idx).tz) == "UTC"


def test_block_rejects_delay_mismatch_and_negative() -> None:
    idx = hourly_index(10)
    frame = pd.DataFrame({"a": [1.0] * 10}, index=idx)
    with pytest.raises(ValueError, match="mismatch"):
        FeatureBlock("toy", frame, {"a": 0, "extra": 1})
    with pytest.raises(ValueError, match="mismatch"):
        FeatureBlock("toy", frame.assign(b=[2.0] * 10), {"a": 0})
    with pytest.raises(ValueError, match=">= 0"):
        FeatureBlock("toy", frame, {"a": -1}, reads_target=True)
    # exogenous block (reads_target=False) must have delay 0
    with pytest.raises(ValueError, match="delay 0"):
        FeatureBlock("exo", frame, {"a": 5})
    with pytest.raises(ValueError, match="timezone-aware"):
        FeatureBlock(
            "toy",
            pd.DataFrame(
                {"a": [1.0] * 10},
                index=pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10)),
            ),
            {"a": 0},
        )


def test_apply_as_of_masks_by_delay() -> None:
    idx = hourly_index(24 * 10)
    block = toy_block(idx, {"det": 0, "lag24": 24})
    as_of = idx[24 * 5]
    out = block.apply_as_of(as_of)
    # delay 0: known only up to as_of itself
    assert out["det"].loc[:as_of].notna().all()
    assert out["det"].loc[as_of + pd.Timedelta(hours=1) :].isna().all()
    # lag24 usable at t iff t - 24h <= as_of, i.e. t <= as_of + 24h
    assert out["lag24"].loc[: as_of + pd.Timedelta(hours=24)].notna().all()
    assert out["lag24"].loc[as_of + pd.Timedelta(hours=25) :].isna().all()


def test_future_perturbation_does_not_leak_into_available_features() -> None:
    """Core anti-leakage test: overwriting the target AFTER ``as_of`` must not
    change any feature that is available at ``as_of`` (delay 24 shields up to
    as_of + 24h). A lag/rolling that forgets its shift fails this test."""
    n = 24 * 30
    idx = hourly_index(n)
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=n).cumsum(), index=idx)
    as_of = idx[24 * 20]

    def make_block(series: pd.Series, *, shift: int) -> FeatureBlock:
        return FeatureBlock(
            "toy_lags",
            series.shift(shift).to_frame("lag_24"),
            {"lag_24": 24},
            reads_target=True,
        )

    y_future = y.copy()
    y_future.loc[as_of + pd.Timedelta(hours=1) :] = 1e9

    base = make_block(y, shift=24).apply_as_of(as_of)
    pert = make_block(y_future, shift=24).apply_as_of(as_of)
    pd.testing.assert_frame_equal(base, pert)

    # broken version (declared delay 24 but no shift): the future leaks in
    base_bad = make_block(y, shift=0).apply_as_of(as_of)
    pert_bad = make_block(y_future, shift=0).apply_as_of(as_of)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(base_bad, pert_bad)


def test_truncation_equivalence_on_past() -> None:
    """On t <= as_of, a feature built from the truncated series equals the
    full build (no border effects)."""
    n = 24 * 30
    idx = hourly_index(n)
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(size=n).cumsum(), index=idx)
    as_of = idx[24 * 20]

    def lag24(series: pd.Series):
        return series.shift(24)

    full = lag24(y).loc[:as_of]
    part = lag24(y.loc[:as_of])
    pd.testing.assert_series_equal(full, part, check_dtype=False)


def test_exogenous_block_always_available() -> None:
    """reads_target=False: known in advance, never masked by as_of."""
    idx = hourly_index(24 * 5)
    block = FeatureBlock(
        "cal", pd.DataFrame({"flag": np.ones(len(idx))}, index=idx), {"flag": 0}
    )
    as_of = idx[24 * 2]
    out = block.apply_as_of(as_of)
    assert out["flag"].notna().all()  # future included
