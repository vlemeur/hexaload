"""Synthetic tests for the monitoring brick (CI-safe: no data/, no network)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hexaload.monitoring import CusumState, compute_status, cusum_update, daily_scores
from hexaload.monitoring.cusum import cusum_series


def test_cusum_no_alarm_on_noise():
    rng = np.random.default_rng(1)
    state = CusumState()
    # h=5, k=0.5 has ARL0 ~ 465: over a short 200-day window, no false alarm
    for x in rng.normal(0, 1, 200):
        state = cusum_update(state, x, k=0.5, h=5.0)
    assert state.n_alarms == 0


def test_cusum_alarms_on_drift_and_resets():
    states, signed = cusum_series([0.0] * 50 + [2.0] * 20, k=0.5, h=5.0)
    assert any(s.alarm for s in states)
    assert states[-1].n_alarms >= 1
    # after an alarm the CUSUM restarts at 0
    alarm_idx = next(i for i, s in enumerate(states) if s.alarm)
    assert states[alarm_idx].s_pos == 0.0 and states[alarm_idx].s_neg == 0.0


def test_daily_scores_and_status(tmp_path):
    # synthetic actuals: 5 days, hourly
    idx = pd.date_range("2026-11-20", periods=24 * 6, freq="h", tz="UTC")
    y: pd.Series = pd.Series(50000.0 + np.arange(len(idx)) % 24, index=idx)
    ref: pd.Series = pd.Series(y.shift(24))
    src: pd.Series = pd.Series("cons-def", index=idx)
    # one published forecast for 2026-11-24 (perfect) and one for 2026-11-25 (bad)
    fdir = tmp_path / "forecasts"
    fdir.mkdir()
    for date_str, offset in (("2026-11-24", 0.0), ("2026-11-25", 5000.0)):
        target = pd.Timestamp(date_str).tz_localize("Europe/Paris").tz_convert("UTC")
        test = pd.date_range(target, periods=24, freq="h", tz="UTC")
        med = (y.reindex(test) + offset).tolist()
        (fdir / f"{date_str}.json").write_text(
            json.dumps(
                {
                    "date": date_str,
                    "as_of": str(test[0] - pd.Timedelta(hours=36)),
                    "models": {"m1": {"median": med}},
                    "rte_d1": y.shift(24).reindex(test).tolist(),
                }
            )
        )
    (fdir / "index.json").write_text(
        json.dumps({"dates": ["2026-11-24", "2026-11-25"]})
    )

    df = daily_scores(fdir, y, ref, source=src)
    assert len(df) == 2
    assert df.loc[df.date == "2026-11-24", "m1__mae"].iloc[0] == pytest.approx(
        0.0, abs=1e-6
    )
    assert df.loc[df.date == "2026-11-25", "m1__mae"].iloc[0] == pytest.approx(5000.0)
    assert "rte_mae" in df.columns
    assert "m1__cusum" in df.columns

    st = compute_status(fdir, rte_available=True)
    assert st["status"] == "served" and st["date"] == "2026-11-25"
    st2 = compute_status(fdir, rte_available=False)
    assert st2["status"] == "fallback" and st2["degraded_inputs"]
    st3 = compute_status(tmp_path / "empty", rte_available=True)
    assert st3["status"] == "none"
