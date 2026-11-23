"""Tests for hexaload.data.fetch — offline: HTTP mocked, cache in tmp_path."""

import json
from datetime import UTC, datetime

import pytest
import requests

from hexaload.data import fetch
from hexaload.data.fetch import (
    FetchError,
    fetch_dataset,
    iter_month_ranges,
    load_manifest,
    read_cache,
)


def _fake_response(payload: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode()
    response.headers["Content-Type"] = "application/json"
    return response


def _mock_session_get(
    monkeypatch: pytest.MonkeyPatch, client: fetch.HttpClient, page: dict
):
    def fake_get(url, params=None, timeout=None):
        return _fake_response(page)

    monkeypatch.setattr(client.session, "get", fake_get)


def test_iter_month_ranges_covers_year_boundary():
    ranges = iter_month_ranges(
        datetime(2022, 11, 15, tzinfo=UTC), datetime(2023, 2, 10, tzinfo=UTC)
    )
    assert [(a.strftime("%Y-%m"), b.strftime("%Y-%m-%d")) for a, b in ranges] == [
        ("2022-12", "2023-01-01"),
        ("2023-01", "2023-02-01"),
        ("2023-02", "2023-02-10"),
    ]


def test_fetch_dataset_writes_parquet_and_manifest(tmp_path, monkeypatch):
    page = {
        "total_count": 2,
        "results": [
            {"date_heure": "2023-01-01T00:15:00+00:00", "consommation": 1},
            {"date_heure": "2023-01-01T00:45:00+00:00", "consommation": 2},
        ],
    }
    client = fetch.HttpClient(min_interval_s=0.0)
    _mock_session_get(monkeypatch, client, page)
    files = fetch_dataset(
        "ds",
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 2, 1, tzinfo=UTC),
        tmp_path,
        client=client,
    )
    assert len(files) == 1
    manifest = load_manifest(tmp_path / "ds")
    assert manifest["ranges"]["2023-01"]["count"] == 2
    assert len(manifest["ranges"]["2023-01"]["sha256"]) == 64
    df = read_cache("ds", tmp_path)
    assert len(df) == 2


def test_fetch_dataset_skips_cached_months(tmp_path, monkeypatch):
    page = {"total_count": 1, "results": [{"x": 1}]}
    client = fetch.HttpClient(min_interval_s=0.0)
    _mock_session_get(monkeypatch, client, page)
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2023, 3, 1, tzinfo=UTC)
    fetch_dataset("ds", start, end, tmp_path, client=client)
    # second run must be served entirely from the cache: count HTTP calls
    calls = {"n": 0}

    def counting_get(url, params=None, timeout=None):
        calls["n"] += 1
        return _fake_response(page)

    monkeypatch.setattr(client.session, "get", counting_get)
    fetch_dataset("ds", start, end, tmp_path, client=client)
    assert calls["n"] == 0  # both months served from cache


def test_read_cache_empty_raises(tmp_path):
    with pytest.raises(FetchError):
        read_cache("nope", tmp_path)


def test_fetch_dataset_progress_bar_smoke(tmp_path, monkeypatch):
    pytest.importorskip("tqdm")
    page = {"total_count": 1, "results": [{"x": 1}]}
    client = fetch.HttpClient(min_interval_s=0.0)
    _mock_session_get(monkeypatch, client, page)
    files = fetch_dataset(
        "ds",
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 3, 1, tzinfo=UTC),
        tmp_path,
        client=client,
        progress=True,
    )
    assert len(files) == 2
