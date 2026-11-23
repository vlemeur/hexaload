"""Tests for hexaload.data.fetch — offline: HTTP mocked, cache in tmp_path."""

import json
import time
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


def _fake_response(
    payload: dict, status: int = 200, headers: dict | None = None
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    response.headers.update({"Content-Type": "application/json", **(headers or {})})
    return response


def _client_with_responses(monkeypatch, responses: list[requests.Response]):
    client = fetch.HttpClient(min_interval_s=0.0)
    state = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        response = responses[min(state["n"], len(responses) - 1)]
        state["n"] += 1
        return response

    monkeypatch.setattr(client.session, "get", fake_get)
    sleeps: list[float] = []
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)
    return client, sleeps


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


def test_429_waits_retry_after_delta(monkeypatch):
    client, sleeps = _client_with_responses(
        monkeypatch,
        [
            _fake_response(
                {"error": True}, status=429, headers={"Retry-After": "0.01"}
            ),
            _fake_response({"ok": True}),
        ],
    )
    assert client.get_json("x", {}) == {"ok": True}
    assert sleeps[0] <= 1.0  # Retry-After is a delta: must not sleep an epoch


def test_throttle_waits_proactively_when_quota_runs_low(monkeypatch):
    client, sleeps = _client_with_responses(
        monkeypatch,
        [
            _fake_response(
                {"ok": 1},
                headers={
                    "X-RateLimit-Remaining": "1",
                    "X-RateLimit-Limit": "100",
                    "X-RateLimit-Reset": str(time.time() + 2),
                },
            ),
            _fake_response({"ok": 2}),
        ],
    )
    assert client.get_json("a", {}) == {"ok": 1}
    assert client.get_json("b", {}) == {"ok": 2}
    # the second call must have waited for the reset window (~3s), not 0s
    assert any(s > 1.5 for s in sleeps)


def test_429_rate_limit_reset_is_epoch_not_delta(monkeypatch):
    client, sleeps = _client_with_responses(
        monkeypatch,
        [
            _fake_response(
                {"error": True},
                status=429,
                headers={"X-RateLimit-Reset": str(time.time() + 0.01)},
            ),
            _fake_response({"ok": True}),
        ],
    )
    assert client.get_json("x", {}) == {"ok": True}
    # misreading the epoch as a delta would sleep ~1.8e9 s here
    assert max(sleeps) < 5


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
