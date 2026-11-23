"""Aggregate per-fold scores into result tables (parquet + markdown).

Outputs are generated artifacts (``results/`` is gitignored): never edit
them by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METRIC_COLUMNS = ["mae", "rmse", "mase"]


def summarize(
    records: list[dict],
    *,
    groupby: tuple[str, ...] = ("model",),
) -> pd.DataFrame:
    """One row per group, with mean/std/count per metric over the folds."""
    df = pd.DataFrame(records)
    metrics = [c for c in METRIC_COLUMNS if c in df.columns]
    agg = (
        df.groupby(list(groupby), sort=True)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [f"{a}_{b}" if b else a for a, b in agg.columns.to_flat_index()]
    return agg


def _to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table without the tabulate dependency."""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join([" --- "] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        cells = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out_parquet: Path, out_md: Path) -> None:
    """Write the table as parquet and as markdown next to it."""
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(df))
