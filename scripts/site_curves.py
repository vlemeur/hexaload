"""Generate ``docs/data/accuracy.json``: a replay of the last 7 days — for each
day D, the same production forecast made at D-1 12:00 Paris (freeze), horizon
24 h, for the foundation model + the classic GBM, against actuals and RTE's
D-1 forecast. Fetched client-side by the site's Accuracy section.

Runs offline on the data/processed cache EXCEPT the live weather for the days
that are not yet in the archive — those use the archive as-of runs (exactly
what a D-1 forecast made that day would have used). ~10-20 min (Timestral on
CPU × 7 days). Regenerate with ``just site-curves``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from hexaload.backtest.load import exogenous_columns, load_features, load_targets
from hexaload.backtest.recursive import (
    block_for,
    build_future_features,
    freeze_as_of,
    recursive_predict,
)
from hexaload.data.fetch import fetch_openmeteo_live_national
from hexaload.features import apply_as_of
from hexaload.metrics import DEFAULT_SP, crps, mase, sample_quantiles
from hexaload.models import DEFAULT_QUANTILES
from hexaload.tuning.params import load_best_params
from hexaload.tuning.spaces import (
    VARIANT_MODELS,
    build_model,
    build_variant,
)

ROOT = Path(__file__).resolve().parent.parent
N_SAMPLES = 100  # empirical CRPS ensemble size (same as backtest/run.py)
OUT = ROOT / "docs/data/accuracy.json"
MODELS = ("timestral_209m", "hgb_features_ewma_mondrian", "ridge_features")
N_DAYS = 7
WINDOW_H = 24 * 365

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    y, ref, wx_hist, _src = load_targets()
    X, manifest = load_features()
    params = load_best_params(ROOT / "results/tuning/best_params.yaml")
    exo = exogenous_columns(manifest)

    today = pd.Timestamp.now(tz="UTC").tz_convert("Europe/Paris").normalize()
    days = [today - pd.Timedelta(days=k) for k in range(1, N_DAYS + 1)][::-1]

    out: dict = {"days": []}
    for target_paris in days:
        target = target_paris.tz_convert("UTC")
        as_of = freeze_as_of(target)
        if as_of >= y.index.max():
            logger.info("skip %s: not enough actuals yet", target.date())
            continue
        test = pd.date_range(target, periods=24, freq="h", tz="UTC")
        block = block_for(as_of, test)  # contiguous as_of+1h .. target+23h
        y_train = y.loc[:as_of].iloc[-WINDOW_H:]
        X_train = apply_as_of(X, manifest, as_of).loc[y_train.index]

        # weather for the target day: archive if covered (as-of run, like a
        # real D-1 forecast that day), else live API as fallback
        wx_all = wx_hist
        if wx_hist.index.max() < test[-1]:
            try:
                wx_live = fetch_openmeteo_live_national(
                    (test[0] - pd.Timedelta(hours=72)).to_pydatetime(),
                    test[-1].to_pydatetime(),
                )
                wx_all = (
                    pd.concat([wx_hist, wx_live]).groupby(level=0).last().sort_index()
                )
            except Exception as exc:
                logger.warning(
                    "live weather failed for %s (%s) — archive only", target.date(), exc
                )
        X_future = build_future_features(
            as_of=as_of,
            block=block,
            y=y,
            wx=wx_all,
            prevision_j1=ref,
            columns=X.columns,
        )

        day: dict = {
            "date": str(target_paris.date()),
            "actual": [float(v) for v in y.reindex(test)],
            "rte_d1": [None if pd.isna(v) else float(v) for v in ref.reindex(test)],
            "models": {},
            "mase": {},
            "crps": {},
        }
        y_test = y.reindex(test)
        for name in MODELS:
            m = (
                build_variant(name, params)
                if name in VARIANT_MODELS
                else build_model(name, params.get(name, {}))
            )
            if name in ("timestral_209m", "chronos_bolt"):
                m.fit(y_train, X_train[exo])
                pred = m.predict(block, X_future[exo]).reindex(test)
                q = m.predict_quantiles(
                    block, X_future[exo], DEFAULT_QUANTILES
                ).reindex(test)
            else:
                m.fit(y_train, X_train)
                # recursive lags: same production freeze (12:00 Paris) as
                # scripts/forecast.py — one-shot predict would NaN all lags on
                # the second half of the target day
                pred_full, q_full = recursive_predict(
                    m,
                    y=y,
                    wx=wx_all,
                    prevision_j1=ref,
                    as_of=as_of,
                    block=block,
                    columns=X.columns,
                    quantiles=DEFAULT_QUANTILES,
                )
                pred = pred_full.reindex(test)
                q = q_full.reindex(test) if q_full is not None else None
            day["models"][name] = {"median": [float(v) for v in pred]}
            if q is not None:
                day["models"][name]["q05"] = [float(v) for v in q["q0.05"]]
                day["models"][name]["q95"] = [float(v) for v in q["q0.95"]]
                samples = sample_quantiles(q, DEFAULT_QUANTILES, N_SAMPLES, seed=0)
                day["crps"][name] = float(crps(y_test, samples))
            day["mase"][name] = float(
                mase(y_test, pred, y_train=y_train, sp=DEFAULT_SP)
            )
            logger.info(
                "%s %s MASE %.3f CRPS %.0f",
                target.date(),
                name,
                day["mase"][name],
                day["crps"].get(name, float("nan")),
            )
        rte_pred = ref.reindex(test)
        if rte_pred.notna().any():
            day["mase"]["rte_prevision_j1"] = float(
                mase(y.reindex(test), rte_pred, y_train=y_train, sp=DEFAULT_SP)
            )
        out["days"].append(day)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out))
    logger.info("wrote %s (%d days)", OUT, len(out["days"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
