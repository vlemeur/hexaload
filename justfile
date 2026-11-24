set default-list := true

# Synchronize runtime and development dependencies from the lockfile
sync:
    uv sync --locked

# Install the development environment and Git hooks
setup: sync
    uv run --locked prek install

# Run the test suite (local, no network)
test:
    uv run --locked pytest

# Test suite with a coverage report (terminal + htmlcov/ + coverage.xml)
test-cov:
    uv run --locked pytest --cov --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml

# Run all Git hooks — the single lint gate, never ruff directly
check:
    uv run --locked prek run --all-files

# Tune one model family (Optuna + MLflow, local SQLite store, offline data).
# Tuning only — use tune-eval for the hold-out afterwards.
# e.g. just tune hgb_features,xgb_features 50 120
tune models="hgb_features,xgb_features" trials="30" budget_minutes="90":
    uv run --locked python scripts/tune.py --mode tune --models "{{models}}" --trials {{trials}} --budget-minutes {{budget_minutes}}

# SMOKE: 1 trial per model, tuning only — validates the whole pipeline.
# torch epochs capped at 10 so even LSTM/TCN finish in minutes: just smoke
smoke budget_minutes="15" max_epochs="10":
    uv run --locked python scripts/tune.py --mode tune --trials 1 --budget-minutes {{budget_minutes}} --max-epochs {{max_epochs}}

# FULL benchmark in ONE command: tune all models -> 2026 hold-out -> backtests
# (tuned + default). Studies are resumable: re-running adds trials.
# M4-friendly defaults: 30 trials, 60 min budget/model (enforced inside trials),
# 3 tuning folds on a 180d window (hold-out + backtests keep the full 365d window).
# Launch in background: nohup just bench > /tmp/hexaload-bench.log 2>&1 &
bench trials="30" budget_minutes="60":
    uv run --locked python scripts/tune.py --mode full --trials {{trials}} --budget-minutes {{budget_minutes}} --folds 3 --tuning-window 4320 --max-epochs 20

# Evaluate the tuned best params on the reserved 2026 hold-out (touched once)
tune-eval:
    uv run --locked python scripts/tune.py --mode eval

# Serve the local MLflow UI (tuning + hold-out runs)
tune-ui:
    uv run --locked mlflow ui --backend-store-uri sqlite:///$PWD/data/mlruns/mlflow.db --port 5000

# Run every quality check
all:
    just test
    just check
