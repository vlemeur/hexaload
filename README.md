# hexaload

[![CI](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml/badge.svg)](https://github.com/vlemeur/hexaload/actions/workflows/ci.yml)

A benchmark of national French electricity load forecasting on RTE's open data (consolidated)


## Layout


## Usage

```bash
uv lock            # one-time: produce uv.lock (network access)
uv sync --locked   # reproducible environment
just setup         # sync + install Git hooks
just check         # the single lint gate: prek run --all-files
```
