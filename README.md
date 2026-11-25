# hexaload

[![CI](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml/badge.svg)](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml)

Benchmark of national French electricity load forecasting on RTE's open data
(eco2mix national, hourly, UTC) — walk-forward only, RTE's own D-1 forecast
as the absolute reference.

**Live site: https://vlemeur.github.io/hexaload/** (results, daily forecast,
7-day accuracy replay, monitoring).

## Headline — 36 walk-forward folds, 2023→2026, D-1 (h=24), tuned

| model | MASE | CRPS | vs RTE |
| --- | ---: | ---: | ---: |
| timestral_209m (zero-shot, exogenous only) | **0.239** | **429** | **−54%** |
| hgb + EWMA + Mondrian (wrappers) | 0.286 | 578 | −45% |
| hgb / xgb (with RTE forecast as input) | 0.311 / 0.307 | 586 / 589 | −40% |
| **rte_prevision_j1 (reference)** | **0.516** | **778** | — |
| seasonal-naive 24h | 1.019 | 2003 | worse |

Hold-out 2026 (11 folds, scored once): timestral **0.161** (CRPS 295), hgb
0.287 (CRPS 451). Without RTE's forecast as input, our GBM still beats RTE
(0.476 vs 0.516). From-scratch deep nets (MLP/RNN/TCN/N-HiTS/PatchTST) do not
beat the GBMs on one year of data — documented on `/method/`.

## Run

```bash
just test                 # pytest, synthetic only (CI-safe)
just check                # prek run --all-files — the only lint gate
just update               # fetch data cache (VPN off) + build features
just bench 30 60          # tune (Optuna+MLflow, resumable) -> hold-out -> backtests
just forecast             # daily D-1 forecast -> docs/data/forecasts/<J>.json
just site-curves          # 7-day D-1 accuracy replay -> docs/data/accuracy.json
just site && just site-serve   # rebuild the site (results.json + zensical) and preview on :8000
```

Backtests run offline on the cache (no network):

```bash
uv run python scripts/run_backtest.py --params results/tuning/best_params.yaml \
  --out results/backtest_tuned.parquet      # D-1, 36 folds, tuned (default: constructor defaults)
uv run python scripts/run_backtest.py --drop-features lag_rte_prevision_j1 \
  --out results/backtest_norte.parquet      # same, without RTE's forecast as input
```

## Protocol

Every forecast is judged by **walk-forward only**: 36 folds, sliding 365-day
train window, 30-day step, horizon h=24 (D-1), 2023→2026 — no random split,
ever. The folds starting in 2026 form a **hold-out scored once**, with the
tuned params.

**Anti-leakage by construction.** Each feature declares a delay (hours of
target history it reads). At each fold origin, target-reading features
(`lag_load_*`, `roll_*`, `lag_rte_prevision_j1`) are masked to NaN wherever
`t − delay > as_of`. Exogenous inputs — calendar and *archived weather
forecasts* — are known in advance; observations never enter, only forecasts
as they were available at forecast time. In production (freeze J-1 12:00
Paris) the 12 h gap before the target day is handled by recursive prediction,
and RTE's not-yet-published forecast is proxied by the 24 h load lag.

**Metrics.** Point errors in MAE / RMSE / MASE (seasonal period 24, denominator
on the train fold); intervals in CRPS over 19 quantile levels (0.05–0.95).
Formulas on the site `/method/` page.

**Tuning.** Optuna NSGA-II multi-objective (MASE + CRPS) on 3 spread folds,
resumable SQLite studies, one MLflow run per trial (both SQLite-backed).
Post-hoc wrappers — EWMA bias correction and Mondrian per-hour conformal
intervals — are part of the tuned objective, with their own hyperparameters.
Best point = min MASE on the Pareto front with CRPS ≤ 1.05×min.

## Layout

```
src/hexaload/
  data/       fetch (network-only) + pydantic validation
  features/   55 leakage-safe columns + manifest + as-of masking
  backtest/   walk-forward engine, recursive production prediction
  models/     ridge/HGB/XGB · AutoETS/Theta · deep torch · TSFM · ensemble/wrappers
  monitoring/ daily scoring + CUSUM drift + status
  tuning/     Optuna spaces/driver + MLflow + hold-out
  metrics/    MAE/RMSE/MASE/CRPS (pure)
scripts/      thin CLIs · docs/ = zensical site (Pages) · .github/workflows = CI + daily forecast
```

`data/` and `results/` are gitignored and reproducible; generated tables are
never hand-edited.
