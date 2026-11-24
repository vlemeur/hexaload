"""Population weights per Open-Meteo grid point (network-free core).

Each 1° grid point (``hexaload.data.fetch.france_grid``) gets the summed
population of the communes whose centroid lies within ``radius_deg`` of the
point (local population weighting — finer than a department match, and the
grid is 1° anyway); points with no commune in range (open sea) fall back to
the nearest commune. Weights are normalized to sum 1 and embedded at
``features/assets/grid_weights.csv`` by ``scripts/build_grid_weights.py``
(network, run yourself) so the backtest runs offline.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_RADIUS_DEG = 0.7  # ~65 km at France's latitude, matches the 1° grid


def assign_grid_weights(
    points: Sequence[tuple[float, float]],
    communes: pd.DataFrame,
    *,
    radius_deg: float = DEFAULT_RADIUS_DEG,
) -> pd.DataFrame:
    """Assign each grid point the population of nearby communes.

    ``communes`` needs columns ``population``, ``centre_lat``, ``centre_lon``.
    Returns ``latitude, longitude, weight`` (weights normalized to sum 1),
    coordinates rounded to 4 decimals like ``france_grid()``.
    """
    required = {"population", "centre_lat", "centre_lon"}
    missing = required - set(communes.columns)
    if missing:
        raise ValueError(f"communes frame missing columns: {sorted(missing)}")
    dep = communes.dropna(subset=["population", "centre_lat", "centre_lon"])
    if dep.empty or dep["population"].sum() <= 0:
        raise ValueError("no positive commune population")
    clat = dep["centre_lat"].to_numpy(float)
    clon = dep["centre_lon"].to_numpy(float)
    pop = dep["population"].to_numpy(float)
    r2 = radius_deg * radius_deg
    rows = []
    for lat, lon in points:
        d2 = (clat - lat) ** 2 + (clon - lon) ** 2  # degrees², fine at 1°
        mask = d2 <= r2
        weight = float(pop[mask].sum()) if mask.any() else float(pop[d2.argmin()])
        rows.append(
            {
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "weight": weight,
            }
        )
    out = pd.DataFrame(rows)
    out["weight"] = out["weight"] / out["weight"].sum()
    return out
