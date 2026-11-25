"""Two-sided CUSUM on a daily standardized gap (model error − reference error).

Pure functions, no I/O. Convention: ``x`` is the daily gap in units of the
in-control standard deviation. ``S+`` accumulates positive drift, ``S-``
negative drift; alarm when |S| > h, then reset (standard CUSUM head-start
restart).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CusumState:
    s_pos: float = 0.0
    s_neg: float = 0.0
    alarm: bool = False
    n_alarms: int = 0


def cusum_update(
    state: CusumState, x: float, *, k: float = 0.5, h: float = 5.0
) -> CusumState:
    """One step of the two-sided CUSUM. Returns a NEW state (pure)."""
    s_pos = max(0.0, state.s_pos + x - k)
    s_neg = min(0.0, state.s_neg + x + k)
    alarm = s_pos > h or s_neg < -h
    if alarm:
        # restart after alarm
        return CusumState(0.0, 0.0, alarm=True, n_alarms=state.n_alarms + 1)
    return CusumState(s_pos, s_neg, alarm=False, n_alarms=state.n_alarms)


def cusum_reset(state: CusumState) -> CusumState:
    return CusumState(0.0, 0.0, alarm=False, n_alarms=state.n_alarms)


def cusum_series(
    xs, *, k: float = 0.5, h: float = 5.0
) -> tuple[list[CusumState], list[float]]:
    """Run the CUSUM over a sequence; returns (states, signed S = S+ + S-)."""
    state = CusumState()
    states: list[CusumState] = []
    signed: list[float] = []
    for x in xs:
        state = cusum_update(state, float(x), k=k, h=h)
        states.append(state)
        signed.append(state.s_pos + state.s_neg)
    return states, signed
