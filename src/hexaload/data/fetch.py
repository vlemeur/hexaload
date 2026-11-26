"""Fetching from open data APIs (ODRE / RTE eco2mix) + eco2mix stitching.

Fetching returns raw records exactly as the APIs serve them. Network
politeness lives here: identified User-Agent, throttling, tenacity retries
with exponential backoff, and honouring rate-limit headers. Every response
is cached on disk as parquet so downstream stages can run fully offline.
``stitch_eco2mix`` (offline) joins the cons-def feed with the real-time tail.

Note: fetch from a network whose exit IP is accepted by the ODRE WAF — some
VPN exit IPs get their TLS handshake reset by the WAF.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

ODRE_BASE = "https://odre.opendatasoft.com/api/explore/v2.1"
DEFAULT_DATASET = "eco2mix-national-cons-def"
PAGE_LIMIT = 100  # hard API cap on this ODRE instance: -1 <= limit <= 100
MIN_REQUEST_INTERVAL_S = 1.0
# A 429 whose rate-limit header announces a wait above this cap fails fast
# instead of sleeping: a CI job must never block for hours on a shared-IP
# quota (Open-Meteo has been seen announcing 80900s from GitHub runners).
# HEXALOAD_MAX_429_WAIT_S overrides it (e.g. forecast.yml's manual cache-seeding
# run waits out Open-Meteo's hourly window instead of failing fast).
MAX_429_WAIT_S = float(os.environ.get("HEXALOAD_MAX_429_WAIT_S", "60"))

# Open-Meteo Historical Forecast API (archived forecasts, replayed as-of date).
OPENMETEO_BASE = "https://historical-forecast-api.open-meteo.com/v1"
OPENMETEO_MODEL = "icon_eu"  # frozen on purpose: a benchmark must be reproducible
# All hourly variables relevant to electricity load (heating/cooling, lighting,
# weather regimes). Frozen list: a benchmark must be reproducible.
OPENMETEO_HOURLY = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,pressure_msl,"
    "precipitation,wind_speed_10m,shortwave_radiation,cloud_cover,"
    "weather_code,is_day"
)
GRID_STEP_DEG = 1.0
GRID_LAT_RANGE = (41.5, 51.5)  # metropolitan France, inclusive
GRID_LON_RANGE = (-5.0, 8.0)
POINTS_PER_REQUEST = 100  # tested OK on the API; halves the request count

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 "
    "hexaload/0.1 (+https://github.com/vlemeur/hexaload)"
)


class FetchError(RuntimeError):
    """A remote call failed after all retries, or the cache is missing."""


def _rate_limit_wait_s(response: requests.Response) -> float:
    """Seconds to wait after a 429, as announced by the server (uncapped).

    Prefer the standard ``Retry-After`` delta; fall back to
    ``X-RateLimit-Reset``. Both headers are sometimes absolute epoch
    timestamps instead of deltas (misreading one sleeps for decades), so
    values above 1e9 are treated as epochs. The caller caps the result at
    ``MAX_429_WAIT_S`` and fails fast beyond it.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            value: float | None = float(retry_after)
        except ValueError:  # HTTP-date form — unsupported, fall through
            value = None
        if value is not None:
            if value > 1e9:  # defensive: absolute epoch, not a delta
                return max(value - time.time(), 1.0)
            return max(value, 1.0)
    try:
        reset = float(response.headers.get("X-RateLimit-Reset", "5"))
    except ValueError:
        reset = 5.0
    if reset > 1e9:  # absolute epoch timestamp
        return max(reset - time.time(), 1.0)
    return max(reset, 1.0)


class HttpClient:
    """Throttled HTTP client with retries and rate-limit awareness."""

    def __init__(
        self,
        base_url: str = ODRE_BASE,
        *,
        user_agent: str = USER_AGENT,
        min_interval_s: float = MIN_REQUEST_INTERVAL_S,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._last_request_at = 0.0
        # server rate-limit state, read from response headers
        self._rl_limit = 0
        self._rl_remaining: int | None = None
        self._rl_reset: float | None = None

    def get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self._get_with_retry(url, params)
        except requests.RequestException as exc:
            raise FetchError(f"GET {url} failed after retries: {exc}") from exc
        return response.json()

    def _throttle(self) -> None:
        # Proactively honour the server rate limit: when the remaining quota
        # runs low, wait for the reset window instead of eating a 429.
        if (
            self._rl_remaining is not None
            and self._rl_limit > 0
            and self._rl_remaining < max(2, int(0.1 * self._rl_limit))
            and self._rl_reset is not None
        ):
            wait = self._rl_reset - time.time() + 1.0
            if wait > MAX_429_WAIT_S:
                # never block for hours on a shared-IP quota; let the request
                # go through and fail fast on the 429 instead
                logger.warning(
                    "rate limit low, reset in %.0fs (> %.0fs cap) — not waiting",
                    wait,
                    MAX_429_WAIT_S,
                )
            elif wait > 0:
                logger.info(
                    "rate limit low (%d/%d left), waiting %.0fs",
                    self._rl_remaining,
                    self._rl_limit,
                    wait,
                )
                time.sleep(wait)
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request_at = time.monotonic()

    def _store_rate_limit_headers(self, response: requests.Response) -> None:
        try:
            self._rl_limit = int(response.headers.get("X-RateLimit-Limit", "0"))
        except ValueError:
            self._rl_limit = 0
        try:
            self._rl_remaining = int(response.headers.get("X-RateLimit-Remaining", ""))
        except ValueError:
            self._rl_remaining = None
        try:
            self._rl_reset = float(response.headers.get("X-RateLimit-Reset", ""))
        except ValueError:
            self._rl_reset = None

    def _get_with_retry(self, url: str, params: dict[str, Any]) -> requests.Response:
        @retry(
            retry=retry_if_exception_type(requests.RequestException),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _do() -> requests.Response:
            self._throttle()
            response = self.session.get(url, params=params, timeout=self.timeout_s)
            self._store_rate_limit_headers(response)
            if response.status_code == 429:
                wait = _rate_limit_wait_s(response)
                if wait > MAX_429_WAIT_S:
                    # fail fast instead of sleeping for hours on a shared-IP
                    # quota; the caller (e.g. the daily forecast cron) retries later
                    raise FetchError(
                        f"HTTP 429 on {url}: rate-limit wait {wait:.0f}s exceeds "
                        f"the {MAX_429_WAIT_S:.0f}s cap — failing fast"
                    )
                logger.warning("429 on %s; waiting %.0fs", url, wait)
                time.sleep(wait)
            if response.status_code >= 400:
                logger.error(
                    "HTTP %s on %s: %s",
                    response.status_code,
                    url,
                    response.text[:500],
                )
            response.raise_for_status()
            return response

        return _do()


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def fetch_records(
    client: HttpClient,
    dataset: str,
    start: datetime,
    end: datetime,
    *,
    select: str | None = None,
    limit: int = PAGE_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """Fetch one page of raw records covering [start, end) (UTC)."""
    where = f'date_heure>="{_iso_utc(start)}" AND date_heure<"{_iso_utc(end)}"'
    params: dict[str, Any] = {"where": where, "limit": limit, "offset": offset}
    if select is not None:
        params["select"] = select
    return client.get_json(f"catalog/datasets/{dataset}/records", params)


def _next_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1)
    return dt.replace(month=dt.month + 1)


def month_key_is_current(key: str, now: datetime | None = None) -> bool:
    """The ongoing UTC month is always stale: it grows every day (the TR tail
    keeps ~85 days, the weather archive gains new issue dates), so it must be
    re-fetched on every update even if its cache file exists."""
    now = now or datetime.now(UTC)
    return key == now.strftime("%Y-%m")


def iter_month_ranges(
    start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into month-aligned UTC ranges (start-inclusive)."""
    cur = start.astimezone(UTC).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    if cur < start:
        cur = _next_month(cur)
    ranges: list[tuple[datetime, datetime]] = []
    while cur < end.astimezone(UTC):
        nxt = min(_next_month(cur), end.astimezone(UTC))
        ranges.append((cur, nxt))
        cur = _next_month(cur)
    return ranges


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def load_manifest(cache_dir: Path | str) -> dict[str, Any]:
    path = _manifest_path(Path(cache_dir))
    if path.exists():
        return json.loads(path.read_text())
    return {"ranges": {}}


def _save_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    _manifest_path(cache_dir).write_text(json.dumps(manifest, indent=2) + "\n")


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> str:
    pd.DataFrame.from_records(records).to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_dataset(
    dataset: str,
    start: datetime,
    end: datetime,
    cache_dir: Path | str,
    *,
    client: HttpClient | None = None,
    refresh: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Fetch [start, end) month by month into ``cache_dir/<dataset>/``.

    Months already present in the manifest are skipped unless ``refresh``.
    Each month lands in ``YYYY-MM.parquet`` (raw records, full schema) and is
    recorded in ``manifest.json`` with its sha256, fetch time and row count.
    Returns the list of cache parquet files covering the range.

    ``progress=True`` shows a tqdm bar over the monthly ranges — meant for
    CLI apps only; the library stays silent (logging) by default.
    """
    client = client or HttpClient()
    cache = Path(cache_dir) / dataset
    cache.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache)
    manifest.setdefault("dataset", dataset)
    files: list[Path] = []
    ranges: Iterable[tuple[datetime, datetime]] = iter_month_ranges(start, end)
    if progress:
        ranges = tqdm(ranges, desc=f"fetch {dataset}", unit="month")
    for month_start, month_end in ranges:
        key = month_start.strftime("%Y-%m")
        path = cache / f"{key}.parquet"
        if (
            not refresh
            and not month_key_is_current(key)
            and manifest.get("ranges", {}).get(key)
            and path.exists()
        ):
            logger.info("cache hit: %s", key)
            files.append(path)
            continue
        records: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = fetch_records(client, dataset, month_start, month_end, offset=offset)
            batch = page.get("results", [])
            records.extend(batch)
            offset += len(batch)
            total = page.get("total_count", len(records))
            if not batch or offset >= total:
                break
        sha256 = _write_parquet(path, records)
        manifest.setdefault("ranges", {})[key] = {
            "file": path.name,
            "sha256": sha256,
            "fetched_at": datetime.now(UTC).isoformat(),
            "count": len(records),
        }
        _save_manifest(cache, manifest)
        files.append(path)
        logger.info("fetched %s: %d records", key, len(records))
    return files


def read_cache(dataset: str, cache_dir: Path | str) -> pd.DataFrame:
    """Read every cached parquet file of a dataset (offline replay).

    Skips ``._*`` AppleDouble artifacts: a tar produced on macOS can carry
    them (xattrs), they match ``*.parquet`` on extraction, and they are not
    parquet files ("Parquet magic bytes not found" on read).
    """
    cache = Path(cache_dir) / dataset
    files = sorted(p for p in cache.glob("*.parquet") if not p.name.startswith("._"))
    if not files:
        raise FetchError(f"no cached parquet files for {dataset!r} in {cache}")
    logger.info("replaying %d cache files for %s", len(files), dataset)
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def france_grid(step: float = GRID_STEP_DEG) -> list[tuple[float, float]]:
    """Regular lat/lon grid over metropolitan France (bounds inclusive)."""
    n_lat = int(round((GRID_LAT_RANGE[1] - GRID_LAT_RANGE[0]) / step)) + 1
    n_lon = int(round((GRID_LON_RANGE[1] - GRID_LON_RANGE[0]) / step)) + 1
    lats = [round(GRID_LAT_RANGE[0] + i * step, 4) for i in range(n_lat)]
    lons = [round(GRID_LON_RANGE[0] + j * step, 4) for j in range(n_lon)]
    return [(lat, lon) for lat in lats for lon in lons]


def fetch_openmeteo_month(
    client: HttpClient,
    points: list[tuple[float, float]],
    start: datetime,
    end: datetime,
    *,
    model: str = OPENMETEO_MODEL,
    hourly: str = OPENMETEO_HOURLY,
) -> list[dict[str, Any]]:
    """Fetch hourly archived forecasts for [start, end) UTC, one packet of points.

    Variables come from ``hourly`` (comma-separated Open-Meteo hourly vars).
    Open-Meteo's ``end_date`` is inclusive; with ``timezone=UTC`` the returned
    times are UTC.
    """
    params: dict[str, Any] = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
        "start_date": start.date().isoformat(),
        "end_date": (end - timedelta(days=1)).date().isoformat(),
        "hourly": hourly,
        "models": model,
        "timezone": "UTC",
    }
    payload = client.get_json("forecast", params)
    if isinstance(payload, dict):  # single-point responses come back bare
        payload = [payload]
    records: list[dict[str, Any]] = []
    for point in payload:
        hourly_data = point.get("hourly", {})
        times = hourly_data.get("time", [])
        var_keys = [k for k in hourly_data if k != "time"]
        for i, t in enumerate(times):
            record: dict[str, Any] = {
                "time": t,
                "latitude": point.get("latitude"),
                "longitude": point.get("longitude"),
                "model": model,
            }
            for key in var_keys:
                record[key] = hourly_data[key][i]
            records.append(record)
    return records


def fetch_openmeteo_dataset(
    start: datetime,
    end: datetime,
    cache_dir: Path | str,
    *,
    model: str = OPENMETEO_MODEL,
    step: float = GRID_STEP_DEG,
    points_per_request: int = POINTS_PER_REQUEST,
    client: HttpClient | None = None,
    refresh: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Fetch the France grid of archived forecasts into the cache, month by month.

    Same cache mechanics as ODRE: ``cache_dir/openmeteo-<model>/YYYY-MM.parquet``
    plus ``manifest.json`` (sha256, fetched_at, count, grid step, model).
    """
    dataset = f"openmeteo-{model}"
    client = client or HttpClient(base_url=OPENMETEO_BASE)
    cache = Path(cache_dir) / dataset
    cache.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache)
    manifest.update({"dataset": dataset, "model": model, "grid_step_deg": step})
    points = france_grid(step)
    packets = [
        points[i : i + points_per_request]
        for i in range(0, len(points), points_per_request)
    ]
    ranges: Iterable[tuple[datetime, datetime]] = iter_month_ranges(start, end)
    if progress:
        ranges = tqdm(ranges, desc=f"fetch {dataset}", unit="month")
    files: list[Path] = []
    for month_start, month_end in ranges:
        key = month_start.strftime("%Y-%m")
        path = cache / f"{key}.parquet"
        if (
            not refresh
            and not month_key_is_current(key)
            and manifest.get("ranges", {}).get(key)
            and path.exists()
        ):
            logger.info("cache hit: %s", key)
            files.append(path)
            continue
        records: list[dict[str, Any]] = []
        for packet in packets:
            records.extend(
                fetch_openmeteo_month(
                    client, packet, month_start, month_end, model=model
                )
            )
        sha256 = _write_parquet(path, records)
        manifest.setdefault("ranges", {})[key] = {
            "file": path.name,
            "sha256": sha256,
            "fetched_at": datetime.now(UTC).isoformat(),
            "count": len(records),
            "points": len(points),
        }
        _save_manifest(cache, manifest)
        files.append(path)
        logger.info(
            "fetched %s: %d records over %d grid points", key, len(records), len(points)
        )
    return files


# --- eco2mix national: automatic cons-def -> tr feed switch -----------------
# ODRE's cons-def feed stops at 2026-08-31 21:45 UTC; the same indicator keeps
# flowing on the "-tr" (temps réel) dataset with a ~85-day sliding retention.
# fetch_eco2mix hides this: callers just give a [start, end) range.
CONS_DEF_DATASET = "eco2mix-national-cons-def"
TR_DATASET = "eco2mix-national-tr"
CONS_DEF_END = datetime(2026, 9, 1, tzinfo=UTC)  # first month with no cons-def data
TR_START = datetime(2026, 8, 31, tzinfo=UTC)  # first day covered by the tr feed


def fetch_eco2mix(
    start: datetime,
    end: datetime,
    cache_dir: Path | str,
    *,
    refresh: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Fetch the national eco2mix series over [start, end), crossing the
    cons-def -> tr feed switch at 2026-08-31 automatically.

    Returns the cache files of both datasets involved.
    """
    files: list[Path] = []
    if start < CONS_DEF_END:
        files += fetch_dataset(
            CONS_DEF_DATASET,
            start,
            min(end, CONS_DEF_END),
            cache_dir,
            refresh=refresh,
            progress=progress,
        )
    if end > TR_START:
        files += fetch_dataset(
            TR_DATASET,
            max(start, TR_START),
            end,
            cache_dir,
            refresh=refresh,
            progress=progress,
        )
    return files


# --- eco2mix stitch ---------------------------------------------------------
# ODRE's ``eco2mix-national-cons-def`` stopped publishing at 2026-08-31 21:45
# UTC. The same indicator keeps flowing on ``eco2mix-national-tr`` ("Données
# temps réel", ~85-day sliding retention), so the national series is:
#   cons-def (corrected) 2012-03 -> 2026-08-31 21:45 UTC
#   + tr     (provisional) 2026-08-31 22:00 UTC -> M-1
# Both feeds share the schema and satisfy the same internal identity
# (consommation = sum(production branches) + ech_physiques + pompage, residual
# ~0.1 % from meter readings + flat allowances), so the stitch does not mix
# two different measurements — only a provisional and a corrected version of
# the same one. The ``source`` column keeps the origin visible downstream.

SOURCE_CONS_DEF = "cons-def"
SOURCE_TR = "tr"


def stitch_eco2mix(cons_def: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the two hourly frames; cons-def wins on overlapping stamps."""
    a = cons_def.copy()
    a["source"] = SOURCE_CONS_DEF
    b = tr.copy()
    b["source"] = SOURCE_TR
    out = pd.concat([a, b])
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


# Live forecast API — used ONLY by scripts/forecast.py at freeze time. The
# archive (above) is keyed by valid time (per-day 00Z runs), so it cannot cover
# future targets; the live API returns the current run, issued before the
# freeze, covering the target horizon. Same model/vars/grid as the archive.
OPENMETEO_LIVE_BASE = "https://api.open-meteo.com/v1"  # HttpClient appends /forecast


def fetch_openmeteo_live_national(
    start: datetime,
    end: datetime,
    *,
    model: str = OPENMETEO_MODEL,
    step: float = GRID_STEP_DEG,
    points_per_request: int = POINTS_PER_REQUEST,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """Current live forecast for the France grid over [start, end] UTC valid
    times, aggregated to the national hourly frame (same pipeline as the
    archive: validate -> population-weighted mean per UTC hour)."""
    from .validate import aggregate_weather_national, validate_weather

    client = client or HttpClient(base_url=OPENMETEO_LIVE_BASE)
    points = france_grid(step)
    records: list[dict[str, Any]] = []
    for i in range(0, len(points), points_per_request):
        packet = points[i : i + points_per_request]
        # end_date is inclusive on the API; add a day so `end` is covered
        records.extend(
            fetch_openmeteo_month(
                client, packet, start, end + timedelta(days=1), model=model
            )
        )
    clean, report = validate_weather(pd.DataFrame(records))
    logger.info(
        "live openmeteo %s: %d raw records, %d valid", model, len(records), len(clean)
    )
    return aggregate_weather_national(clean)
