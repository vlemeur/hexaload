"""Daily D-1 forecast for tomorrow (idempotent) — `just forecast`.

Single fit per model on data available at **J-1 12:00 Paris** (the freeze time,
like the reference dashboard): train window = last 365 days before the freeze.
Writes ``docs/data/forecasts/<J>.json`` with, per model, the D-1 (h=24) median +
90 % band (q05/q95) and the D+7 (h=168) median curve. The site's Forecast charts
fetch this file client-side — a JSON push updates the site without a rebuild.

Re-running for the same J is a no-op unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from hexaload.backtest.load import exogenous_columns, load_features, load_targets
from hexaload.backtest.recursive import recursive_predict
from hexaload.features import apply_as_of
from hexaload.models import DEFAULT_QUANTILES
from hexaload.progress import setup_logging
from hexaload.tuning.params import load_best_params
from hexaload.tuning.spaces import (
    FEATURE_MODELS,
    TSFM_MODELS,
    VARIANT_MODELS,
    build_model,
    build_variant,
)

PARIS = ZoneInfo("Europe/Paris")
OUT_DIR = Path("docs/data/forecasts")
WINDOW_H = 24 * 365
FREEZE_HOUR = 12  # Paris time, day before the target
DEFAULT_MODELS = ("timestral_209m", "hgb_features_ewma_mondrian")

logger = logging.getLogger(__name__)


def default_target(now: pd.Timestamp) -> pd.Timestamp:
    """Forecast target (Paris midnight, tz=Paris): tomorrow if before noon
    Paris time, else the day after tomorrow."""
    now_paris = now.tz_convert(PARIS)
    return now_paris.normalize() + pd.Timedelta(
        days=1 if now_paris.hour < FREEZE_HOUR else 2
    )


def block_for(as_of: pd.Timestamp, test_idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Contiguous model block from as_of+1h through the end of ``test_idx``
    (freeze at 12:00 Paris, target day starts at 00:00 Paris — the model must
    predict the contiguous block after the context, then we slice the target
    day out of it)."""
    start = as_of + pd.Timedelta(hours=1)
    n = int((test_idx[-1] - start) / pd.Timedelta(hours=1)) + 1
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def freeze_as_of(target_utc: pd.Timestamp) -> pd.Timestamp:
    """Data freeze for the forecast of ``target``: J-1 at 12:00 Paris, in UTC."""
    j_paris = target_utc.tz_convert(PARIS)
    freeze = (j_paris - pd.Timedelta(days=1)).normalize().replace(hour=FREEZE_HOUR)
    return freeze.tz_convert("UTC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=None,
        help="target day J, YYYY-MM-DD (Paris). Default: tomorrow if before "
        f"{FREEZE_HOUR}:00 Paris, else the day after.",
    )
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="comma-separated model names"
    )
    parser.add_argument(
        "--params",
        default="results/tuning/best_params.yaml",
        help="tuned params YAML (default: the benchmark best params)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing JSON"
    )
    parser.add_argument(
        "--no-live-weather",
        dest="live_weather",
        action="store_false",
        help="do not call the live Open-Meteo API (offline tests / archive-only)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    target_paris = (
        pd.Timestamp(args.date).tz_localize(PARIS)
        if args.date
        else default_target(pd.Timestamp.now(tz="UTC"))
    )
    target = target_paris.tz_convert("UTC")
    date_label = str(target_paris.date())
    as_of = freeze_as_of(target)
    out_path = OUT_DIR / f"{date_label}.json"
    y, ref, wx_hist, _src = load_targets()
    if out_path.exists() and not args.force:
        logger.info("%s already exists — skip (use --force to overwrite)", out_path)
        index_path = OUT_DIR / "index.json"
        if not index_path.exists():
            dates = sorted(
                {p.stem for p in OUT_DIR.glob("*.json") if p.name != "index.json"}
            )
            index_path.write_text(json.dumps({"dates": dates}))
        # RTE publishes prevision_j1 for J later on J-1: fill it in without
        # re-running the models when it becomes available
        payload = json.loads(out_path.read_text())
        test_d1 = pd.date_range(target, periods=24, freq="h", tz="UTC")
        rte_now = [None if pd.isna(v) else float(v) for v in ref.reindex(test_d1)]
        if any(v is not None for v in rte_now) and all(
            v is None for v in payload.get("rte_d1", [])
        ):
            payload["rte_d1"] = rte_now
            out_path.write_text(json.dumps(payload))
            logger.info("filled rte_d1 for %s (RTE published)", date_label)
        return 0

    logger.info(
        "forecast for %s | freeze (as_of) %s UTC | models %s",
        date_label,
        as_of,
        args.models,
    )
    X, manifest = load_features()
    params = load_best_params(args.params) if args.params else {}
    if X is None or manifest is None:
        raise SystemExit("no feature table — run `just update` first")
    exo = exogenous_columns(manifest)

    y_train_all = y.loc[:as_of]
    y_train = y_train_all.iloc[-WINDOW_H:]
    X_train = apply_as_of(X, manifest, as_of).loc[y_train.index]

    test_d1 = pd.date_range(target, periods=24, freq="h", tz="UTC")
    test_full = pd.date_range(target, periods=168, freq="h", tz="UTC")
    block = block_for(as_of, test_full)  # contiguous as_of+1h .. target+168h

    # Future exogenous features: the ARCHIVE is keyed by valid time (per-day 00Z
    # runs), so it cannot cover the target day at freeze time — fetch the LIVE
    # forecast (current run, issued before the freeze) and build the future
    # feature table with the same blocks + as_of mask as the backtest.
    if args.live_weather:
        from hexaload.data.fetch import fetch_openmeteo_live_national

        try:
            # start 72 h before the target: the archive lags ~1-2 days behind,
            # and weather features with delay 12/24 h at the start of the D-1
            # block would otherwise fall into the hole between archive and live
            wx_live = fetch_openmeteo_live_national(
                (test_full[0] - pd.Timedelta(hours=72)).to_pydatetime(),
                test_full[-1].to_pydatetime(),
            )
            wx_all = pd.concat([wx_hist, wx_live]).groupby(level=0).last().sort_index()
            logger.info(
                "live weather: %d hours, %s → %s",
                len(wx_live),
                wx_live.index.min(),
                wx_live.index.max(),
            )
        except Exception as exc:
            logger.warning("live weather fetch failed (%s) — archive only", exc)
            wx_all = wx_hist
    else:
        wx_all = wx_hist
    from hexaload.features.table import build_feature_table

    # build over a padded index so weather lags/gradients have history, then
    # slice to the forecast range
    idx_build = pd.date_range(
        block[0] - pd.Timedelta(hours=72),
        periods=len(block) + 72,
        freq="h",
        tz="UTC",
    )
    X_build, manifest_f = build_feature_table(
        idx_build, as_of=as_of, wx=wx_all, y=y, prevision_j1=ref, horizon=24
    )
    X_future = X_build.loc[block]
    missing = [c for c in X.columns if c not in X_future.columns]
    if missing:
        raise SystemExit(
            f"future feature table misses columns {missing[:5]} — rebuild the "
            "feature table (`just update`) so train/future schemas match"
        )
    X_future = X_future[X.columns]
    n_nan_exo = int(X_future.loc[test_d1, exo].isna().sum().sum())
    if n_nan_exo:
        raise SystemExit(
            f"{n_nan_exo} NaN exogenous cells over the D-1 block — live weather did not"
            "cover it; check the network / Open-Meteo, or run with --no-live-weather "
            "after `just update`"
        )
    # D+7 curve clipped to the exogenous coverage (live run lead time)
    covered = int(
        X_future[exo].notna().all(axis=1).reindex(test_full).ffill().fillna(False).sum()
    )
    n_d7 = max(24, min(168, covered))
    test_d7 = pd.date_range(target, periods=n_d7, freq="h", tz="UTC")

    payload: dict = {
        "date": date_label,
        "as_of": str(as_of),
        "horizon_h": 24,
        "horizon_long_h": len(test_d7),
        "models": {},
        "rte_d1": [None if pd.isna(v) else float(v) for v in ref.reindex(test_d1)],
    }
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        m = (
            build_variant(name, params)
            if name in VARIANT_MODELS
            else build_model(name, params.get(name, {}))
        )
        if name in TSFM_MODELS:
            m.fit(y_train, X_train[exo] if X_train is not None else None)
            pred = m.predict(block, X_future.loc[block][exo]).reindex(test_d1)
            q = m.predict_quantiles(
                block, X_future.loc[block][exo], DEFAULT_QUANTILES
            ).reindex(test_d1)
            pred_d7 = m.predict(block, X_future.loc[block][exo]).reindex(test_d7)
        elif name in FEATURE_MODELS or name in VARIANT_MODELS:
            if X_train is None:
                raise ValueError(f"{name} needs the feature table")
            m.fit(y_train, X_train)
            # recursive lags: the freeze (12:00 Paris) leaves a 12 h gap before
            # the target day, so one-shot predict would see all lags NaN on the
            # afternoon/evening peak — predict the block hour by hour instead
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
            pred = pred_full.reindex(test_d1)
            q = q_full.reindex(test_d1)
            pred_d7 = pred_full.reindex(test_d7)
        else:
            m.fit(y_train)
            pred = m.predict(block).reindex(test_d1)
            q = m.predict_quantiles(block, quantiles=DEFAULT_QUANTILES).reindex(test_d1)
            pred_d7 = m.predict(block).reindex(test_d7)
        payload["models"][name] = {
            "median": [float(v) for v in pred],
            "q05": [float(v) for v in q["q0.05"]],
            "q95": [float(v) for v in q["q0.95"]],
            "median_d7": [float(v) for v in pred_d7],
        }
        logger.info("%s done (D-1 median[0]=%.0f MW)", name, pred.iloc[0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    # index of published dates (the site picks the latest)
    index_path = OUT_DIR / "index.json"
    dates = sorted({p.stem for p in OUT_DIR.glob("*.json") if p.name != "index.json"})
    index_path.write_text(json.dumps({"dates": dates}))
    logger.info("wrote %s (index: %d dates)", out_path, len(dates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
