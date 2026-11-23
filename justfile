set default-list := true

# Synchronize runtime and development dependencies from the lockfile
sync:
    uv sync --locked

# Install the development environment and Git hooks
setup: sync
    uv run --locked prek install

# Run all Git hooks — the single lint gate, never ruff directly
check:
    uv run --locked prek run --all-files

# Run every quality check
all:
    just check

# test / test-cov recipes return with the first tests (phase 3):
#     uv run --locked pytest
#     uv run --locked pytest --cov --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml
