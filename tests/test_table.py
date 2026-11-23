"""Tests for the results table aggregation (offline)."""

from pathlib import Path

import pandas as pd

from hexaload.results.table import summarize, write_table

RECORDS = [
    {"fold": "f1", "model": "a", "mae": 1.0, "rmse": 2.0, "mase": 0.5},
    {"fold": "f2", "model": "a", "mae": 3.0, "rmse": 4.0, "mase": 1.5},
    {"fold": "f1", "model": "b", "mae": 10.0, "rmse": 10.0, "mase": 2.0},
]


def test_summarize_aggregates_per_model() -> None:
    table = summarize(RECORDS)
    assert list(table["model"]) == ["a", "b"]
    row_a = table[table["model"] == "a"].iloc[0]
    assert row_a["mae_mean"] == 2.0
    assert row_a["mae_count"] == 2
    assert row_a["rmse_mean"] == 3.0
    assert row_a["mase_mean"] == 1.0


def test_write_table_parquet_and_markdown(tmp_path: Path) -> None:
    table = summarize(RECORDS)
    out_pq = tmp_path / "results" / "backtest.parquet"
    out_md = tmp_path / "results" / "backtest.md"
    write_table(table, out_pq, out_md)
    loaded = pd.read_parquet(out_pq)
    pd.testing.assert_frame_equal(loaded, table)
    md = out_md.read_text()
    assert "| model |" in md
    assert "mae_mean" in md
