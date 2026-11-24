"""Grid-weight assignment (network-free core of build_grid_weights.py)."""

import pandas as pd
import pytest

from hexaload.features.grid_weights import assign_grid_weights


def test_population_summed_within_radius() -> None:
    communes = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "population": [1_000.0, 2_000.0, 5_000_000.0],
            "centre_lat": [48.0, 48.3, 43.3],  # A,B near point; C far (Marseille)
            "centre_lon": [2.0, 2.2, 5.4],
        }
    )
    points = [(48.1, 2.1), (43.3, 5.4)]
    w = assign_grid_weights(points, communes, radius_deg=0.7)
    assert list(w.columns) == ["latitude", "longitude", "weight"]
    assert w["weight"].sum() == pytest.approx(1.0)
    # point 1 gets A+B = 3000 ; point 2 gets C = 5e6
    assert w.loc[0, "weight"] == pytest.approx(3_000 / 5_003_000)
    assert w.loc[1, "weight"] == pytest.approx(5_000_000 / 5_003_000)


def test_sea_point_falls_back_to_nearest_commune() -> None:
    communes = pd.DataFrame(
        {
            "code": ["A"],
            "population": [1_000.0],
            "centre_lat": [48.0],
            "centre_lon": [2.0],
        }
    )
    w = assign_grid_weights([(42.0, -2.0)], communes, radius_deg=0.7)  # Atlantic
    assert w.loc[0, "weight"] == pytest.approx(1.0)  # nearest (only) commune


def test_bad_input_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        assign_grid_weights([(48.0, 2.0)], pd.DataFrame({"population": [1.0]}))
    with pytest.raises(ValueError, match="no positive"):
        assign_grid_weights(
            [(48.0, 2.0)],
            pd.DataFrame(
                {"population": [None], "centre_lat": [48.0], "centre_lon": [2.0]}
            ),
        )
