# hexaload

[![CI](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml/badge.svg)](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml)

A benchmark of national French electricity load forecasting on RTE's open data (consolidated)


## Layout

- `src/hexaload/data/` — network-only fetching (ODRE/RTE eco2mix) with an
  on-disk parquet cache, and strict, network-free validation (pydantic v2);
- `scripts/` — application entrypoints (fetch, backtest, …);
- `data/` — cache + processed data (gitignored; the backtest replays offline).

## Data acquisition

Fetch the eco2mix national consolidated dataset into `data/cache/` and
validate it into an hourly series under `data/processed/`:

```bash
# VPN OFF when fetching: the ODRE WAF resets TLS handshakes from some VPN
# exit IPs
uv run python scripts/fetch_data.py --from 2012-02-22 --to 2026-12-01
uv run python scripts/fetch_data.py --offline   # replay from cache only
```

## Usage

```bash
uv lock            # one-time: produce uv.lock (network access)
uv sync --locked   # reproducible environment
just setup         # sync + install Git hooks
just check         # the single lint gate: prek run --all-files
```
