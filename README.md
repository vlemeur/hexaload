# hexaload

[![CI](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml/badge.svg)](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml)

A benchmark of national French electricity load forecasting on RTE's open data (consolidated)


## Layout

- `src/hexaload/data/` — network-only fetching (ODRE/RTE eco2mix + Open-Meteo
  weather archive) with an on-disk parquet cache, and strict, network-free
  validation (pydantic v2);
- `scripts/` — application entrypoints (data fetch, backtest);
- `notebooks/` — exploratory data analysis (plotly);
- `data/` — cache + processed data (gitignored; `--offline` replays it).

## Data acquisition

Fetch the open datasets (eco2mix national + Open-Meteo weather archive) into
`data/cache/` and validate them into hourly series under `data/processed/`:

```bash
uv run python scripts/fetch_data.py --from 2023-01-01 --to 2026-12-01
uv run python scripts/fetch_data.py --offline   # replay from cache only
```

## Backtest

Offline walk-forward evaluation (naive baselines + RTE's own D-1 forecast as
reference) over the processed hourly series; writes `results/backtest.parquet`
+ `.md`:

```bash
uv run python scripts/run_backtest.py
```

## Usage

```bash
uv sync --locked   # reproducible environment (uv.lock is committed)
just setup         # sync + install Git hooks
just test          # test suite (local, no network)
just check         # the single lint gate: prek run --all-files
```
