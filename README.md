# hexaload

[![CI](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml/badge.svg)](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml)

Benchmark of national French electricity load forecasting on RTE's open data
(eco2mix national, hourly, UTC). Walk-forward — no random split, ever.
RTE's own D-1 forecast is the absolute reference.

## Headline — 36 walk-forward folds, 2023→2026, D-1 (h=24)

| model | MASE | CRPS | vs RTE |
| --- | ---: | ---: | ---: |
| timestral_209m (zero-shot TSFM) | **0.237** | **426** | −54 % |
| hgb_features | 0.315 | 596 | −39 % |
| xgb_features | 0.315 | 601 | −39 % |
| ridge_features | 0.348 | 620 | −33 % |
| chronos_bolt (zero-shot TSFM) | 0.399 | 763 | −23 % |
| **rte_prevision_j1 (reference)** | **0.516** | **778** | — |
| sktime_autoets / theta | 0.90 / 1.15 | 1685 / 2188 | worse |
| seasonal-naive 24 h / 168 h / naive | 1.02 / 1.29 / 2.15 | 2003 / 2667 / 4590 | worse |

Hold-out 2026 (11 folds, scored once, tuned params): timestral **0.156**,
hgb 0.291, xgb 0.294, ridge 0.331, chronos 0.336 (CRPS 289–626).

Deep models trained from scratch (MLP, LSTM/GRU, TCN, PatchTST, N-HiTS) were
tuned (Optuna NSGA-II, 30 trials each) and **do not beat the GBMs**
on this task (best MASE 0.32 for N-HiTS, up to 2.79 for MLP) — write-up
with the Pareto fronts on the site `/method/` page. Full 36-fold backtest with
torch models is a documented follow-up.

## Run

```bash
just test            # pytest, synthetic data only (CI-safe, no network)
just check           # prek run --all-files — the only lint gate
just bench 30 60     # one command: tune all (Optuna+MLflow) -> 2026 hold-out -> backtests
```

Backtests (offline, read the `data/processed/` cache, no network):

```bash
uv run python scripts/run_backtest.py --params results/tuning/best_params.yaml \
  --out results/backtest_tuned.parquet            # D-1, tuned
uv run python scripts/run_backtest.py --out results/backtest_default.parquet
uv run python scripts/run_backtest.py --horizon 168 --end 2026-11-22 \
  --params results/tuning/best_params.yaml --out results/backtest_d7.parquet   # D+7
```

## Protocol (short version)

- 36 walk-forward folds: sliding 365 d train window, 30 d step, h=24,
  2023-01-01 → 2026-11-24; the 2026 folds are a hold-out scored once.
- Anti-leakage: target-reading features are masked at the fold origin
  (delay-aware as-of mask); only archived weather *forecasts* enter as
  exogenous inputs, never observations. Intervals scored in CRPS
  (19 quantile levels), point errors in MAE/RMSE/MASE (sp=24).
- Tuning: Optuna NSGA-II multi-objective (MASE + CRPS) on 3 folds × 180 d,
  resumable SQLite studies (`data/tuning/optuna.db`), MLflow SQLite backend
  (`data/mlruns/mlflow.db`); best params → `results/tuning/best_params.yaml`.

## Data

RTE eco2mix national (`cons-def` stitched with the real-time `tr` tail) +
Open-Meteo historical-forecast archive. `data/` is gitignored — fetch once
(VPN off: the ODRE WAF resets TLS from some exit IPs), then everything runs
offline. `results/` is generated too — never hand-edit the tables.

## Layout

```
src/hexaload/
  data/       network-only fetching + validation (pydantic)
  features/   leakage-safe feature blocks + manifest (55 cols) + as-of masking
  backtest/   walk-forward engine, fold masking, baselines (RTE included)
  models/     ridge/HGB/XGB · AutoETS/Theta · MLP/RNN/TCN/N-HiTS/PatchTST · Timestral/Chronos
  tuning/     Optuna spaces + driver + MLflow tracking + hold-out eval
  metrics/    MAE/RMSE/MASE/CRPS (pure, no network)
  results/    generated summary tables
scripts/      thin CLIs (build_features, run_backtest, tune) · justfile: test/check/bench
```
