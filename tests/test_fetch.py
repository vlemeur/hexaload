"""Tests for hexaload.data.fetch — offline: HTTP mocked, cache in tmp_path."""

import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import requests

from hexaload.data import fetch
from hexaload.data.fetch import (
    CONS_DEF_DATASET,
    CONS_DEF_END,
    SOURCE_CONS_DEF,
    SOURCE_TR,
    TR_DATASET,
    TR_START,
    FetchError,
    fetch_dataset,
    iter_month_ranges,
    load_manifest,
    read_cache,
    stitch_eco2mix,
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


def _hourly_frame(start: str, n: int, base: float) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "consommation": base + np.arange(n, dtype=float),
            "prevision_j1": base + 100 + np.arange(n, dtype=float),
        },
        index=idx.rename("date_heure_utc"),
    )


def test_stitch_appends_tr_after_cons_def() -> None:
    cons = _hourly_frame("2026-08-31T18:00:00+00:00", 4, 40_000.0)  # ..21:00
    tr = _hourly_frame("2026-08-31T22:00:00+00:00", 3, 39_000.0)  # 22:00..
    out = stitch_eco2mix(cons, tr)
    assert len(out) == 7
    assert list(out["source"]) == [SOURCE_CONS_DEF] * 4 + [SOURCE_TR] * 3
    assert out.index.is_monotonic_increasing
    assert out["consommation"].iloc[-1] == 39_002.0


def test_stitch_cons_def_wins_on_overlap() -> None:
    cons = _hourly_frame("2026-08-31T20:00:00+00:00", 4, 40_000.0)  # ..23:00
    tr = _hourly_frame("2026-08-31T22:00:00+00:00", 2, 99_000.0)  # overlap 22,23
    out = stitch_eco2mix(cons, tr)
    assert len(out) == 4
    assert (out["source"] == SOURCE_CONS_DEF).all()
    assert out["consommation"].iloc[-1] == 40_003.0  # not the tr value


def test_stitch_empty_tr() -> None:
    cons = _hourly_frame("2026-08-31T20:00:00+00:00", 2, 40_000.0)
    out = stitch_eco2mix(cons, cons.iloc[:0])
    assert len(out) == 2
    assert (out["source"] == SOURCE_CONS_DEF).all()


def test_fetch_eco2mix_splits_at_feed_switch(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_fetch_dataset(dataset, start, end, cache_dir, **kwargs):
        calls.append((dataset, start, end))
        return []

    monkeypatch.setattr(fetch, "fetch_dataset", fake_fetch_dataset)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2026, 11, 24, tzinfo=UTC)
    fetch.fetch_eco2mix(start, end, tmp_path)
    assert calls == [
        (CONS_DEF_DATASET, start, CONS_DEF_END),
        (TR_DATASET, TR_START, end),
    ]


def test_fetch_eco2mix_old_range_only_cons_def(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        fetch, "fetch_dataset", lambda *a, **k: calls.append(a[0]) or []
    )
    fetch.fetch_eco2mix(
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC), tmp_path
    )
    assert calls == [CONS_DEF_DATASET]


def test_fetch_eco2mix_recent_range_only_tr(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        fetch, "fetch_dataset", lambda *a, **k: calls.append(a[0]) or []
    )
    fetch.fetch_eco2mix(
        datetime(2026, 9, 15, tzinfo=UTC), datetime(2026, 11, 24, tzinfo=UTC), tmp_path
    )
    assert calls == [TR_DATASET]
