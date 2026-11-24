"""school_calendar module: resource discovery (HTTP mocked) + embed (offline)."""

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from hexaload.data import fetch
from hexaload.features.school_calendar import (
    DATA_GOUV_API,
    embed_school_calendar,
    find_csv_resource_url,
)


def _fake_response(payload: dict, status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    response.headers.update({"Content-Type": "application/json"})
    return response


def test_find_csv_resource_url_uses_data_gouv_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_urls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        seen_urls.append(url)
        return _fake_response(
            {
                "resources": [
                    {"title": "Dépôt GitHub", "format": "git", "url": "https://x"},
                    {
                        "title": "vacances.csv",
                        "format": "csv/utf8",
                        "url": "https://raw.example/vacances.csv",
                    },
                ]
            }
        )

    client = fetch.HttpClient(base_url=DATA_GOUV_API, min_interval_s=0.0)
    monkeypatch.setattr(client.session, "get", fake_get)
    url = find_csv_resource_url(client)
    assert url == "https://raw.example/vacances.csv"
    # must hit the data.gouv.fr API, NOT the ODRE base
    assert seen_urls == [f"{DATA_GOUV_API}/datasets/vacances-scolaires-par-zones/"]


def test_find_csv_resource_url_raises_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = fetch.HttpClient(base_url=DATA_GOUV_API, min_interval_s=0.0)
    monkeypatch.setattr(
        client.session, "get", lambda *a, **k: _fake_response({"resources": []})
    )
    with pytest.raises(RuntimeError, match="no vacances CSV resource"):
        find_csv_resource_url(client)


def test_embed_school_calendar_validates_and_filters(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2011-06-01",
                "vacances_zone_a": True,
                "vacances_zone_b": True,
                "vacances_zone_c": True,
                "nom_vacances": "été",
            },
            {
                "date": "2025-10-20",
                "vacances_zone_a": "true",
                "vacances_zone_b": False,
                "vacances_zone_c": False,
                "nom_vacances": "Toussaint",
            },
            {
                "date": "bogus",
                "vacances_zone_a": True,
                "vacances_zone_b": True,
                "vacances_zone_c": True,
                "nom_vacances": "x",
            },
        ]
    )
    out = tmp_path / "embedded.csv"
    clean = embed_school_calendar(raw, out)
    # 2011 row filtered out (EMBED_START), bogus row rejected by pydantic
    assert len(clean) == 1
    assert clean.index[0] == pd.Timestamp("2025-10-20")
    assert bool(clean["vacances_zone_a"].iloc[0]) is True
    reread = pd.read_csv(out)
    assert list(reread.columns) == [
        "date",
        "vacances_zone_a",
        "vacances_zone_b",
        "vacances_zone_c",
        "nom_vacances",
    ]
    assert len(reread) == 1
