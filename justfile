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

# Tune one model family (Optuna + MLflow, local SQLite/file store, offline data)
# e.g. just tune hgb_features,xgb_features 50 120
tune models="hgb_features,xgb_features" trials="30" budget_minutes="90":
    uv run --locked python scripts/tune.py --models "{{models}}" --trials {{trials}} --budget-minutes {{budget_minutes}}

# FULL benchmark in ONE command: tune all models -> 2026 hold-out -> backtests
# (tuned + default). Studies are resumable: re-running adds trials.
# e.g. just bench 50 180   (50 trials/model, up to 180 min budget per model)
bench trials="50" budget_minutes="180":
    uv run --locked python scripts/tune.py --mode full --trials {{trials}} --budget-minutes {{budget_minutes}}

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
