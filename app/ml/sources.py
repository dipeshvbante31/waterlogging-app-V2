"""Download helpers for the two verified data sources used by ``scripts/prepare_dataset.py``.

* Open-Meteo Historical Weather API (hourly, ERA5-based reanalysis). No API key.
  Data licence CC BY 4.0; the free endpoint is for NON-COMMERCIAL use only.
* NYC OpenData "311 Service Requests" via the public Socrata (SoQL) API. An app
  token is optional (it only raises rate limits).

Responses are cached under ``data/raw`` so a re-run does not hit the providers again.
"""
from __future__ import annotations

import time
import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import httpx
import pandas as pd

from app.ml.dataset import WEATHER_COLUMNS, weather_frame

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SOCRATA_URL = "https://data.cityofnewyork.us/resource/{rid}.json"
NYC_311_2010_2019 = "76ig-c548"
NYC_311_2020_PRESENT = "erm2-nwe9"

NYC_BOROUGH_POINTS = {   # approximate borough centres (configuration, not measured data)
    "MANHATTAN": (40.7831, -73.9712),
    "BROOKLYN": (40.6782, -73.9442),
    "QUEENS": (40.7282, -73.7949),
    "BRONX": (40.8448, -73.8648),
    "STATEN ISLAND": (40.5795, -74.1502),
}


def _get_json(client: httpx.Client, url: str, params: dict, log: Callable[[str], None], retries: int = 6):
    last = ""
    for attempt in range(retries):
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError as exc:
            last = f"network error: {exc}"
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
            elif resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:500]}")
            else:
                return resp.json()
        wait = min(90, 3 * 2 ** attempt)
        log(f"  {last}; retrying in {wait}s ({attempt + 1}/{retries})")
        time.sleep(wait)
    raise RuntimeError(f"Giving up on {url}: {last}")


def fetch_weather(client: httpx.Client, lat: float, lon: float, start: date, end: date,
                  cache_dir: Path, log: Callable[[str], None], model: str | None = None) -> pd.DataFrame:
    """Hourly UTC weather for one point, fetched one calendar year at a time."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(start.year, end.year + 1):
        a, b = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        cache = cache_dir / f"weather_{lat:.4f}_{lon:.4f}_{a}_{b}{'_' + model if model else ''}.csv"
        if cache.exists():
            frames.append(pd.read_csv(cache, index_col=0, parse_dates=True))
            continue
        params = {"latitude": lat, "longitude": lon, "start_date": a.isoformat(), "end_date": b.isoformat(),
                  "hourly": ",".join(WEATHER_COLUMNS)}
        if model:
            params["models"] = model
        log(f"  weather {lat:.3f},{lon:.3f} {a}..{b}")
        data = _get_json(client, ARCHIVE_URL, params, log)
        hourly = data.get("hourly") or {}
        if "time" not in hourly or any(c not in hourly for c in WEATHER_COLUMNS):
            raise RuntimeError(f"Unexpected archive response keys: {list(hourly)}")
        frame = weather_frame(hourly["time"], {c: hourly[c] for c in WEATHER_COLUMNS})
        frame.to_csv(cache)
        frames.append(frame)
        time.sleep(1.0)
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]


def month_ranges(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        lo = max(start, date(y, m, 1))
        hi_excl = date(ny, nm, 1)
        yield lo, hi_excl
        y, m = ny, nm


def resource_for(day: date) -> str:
    return NYC_311_2010_2019 if day.year <= 2019 else NYC_311_2020_PRESENT


def list_flood_descriptors(client: httpx.Client, start: date, end: date, log, app_token: str | None = None):
    """Diagnostics: which complaint/descriptor values mention 'flood' (verify the filter before a long download)."""
    rows = []
    for rid in sorted({resource_for(start), resource_for(end)}):
        params = {"$select": "complaint_type,descriptor,count(*) AS n",
                  "$where": f"lower(descriptor) like '%flood%' AND created_date >= '{start}T00:00:00' "
                            f"AND created_date < '{end}T00:00:00'",
                  "$group": "complaint_type,descriptor", "$limit": 100}
        if app_token:
            params["$$app_token"] = app_token
        for r in _get_json(client, SOCRATA_URL.format(rid=rid), params, log):
            rows.append({"resource": rid, **r})
    return rows


def fetch_nyc311_counts(client: httpx.Client, start: date, end: date, complaint_type: str, descriptor_like: str,
                        cache_dir: Path, log, app_token: str | None = None) -> pd.DataFrame:
    """Hourly (local-time) incident counts per borough for the selected 311 complaints.

    Returns columns: local_time (naive, America/New_York wall clock), borough, n.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for lo, hi in month_ranges(start, end):
        hi = min(hi, end + timedelta(days=1))
        rid = resource_for(lo)
        tag = hashlib.sha1(f"{complaint_type}|{descriptor_like}".encode()).hexdigest()[:8]
        cache = cache_dir / f"nyc311_{rid}_{tag}_{lo:%Y-%m}.csv"
        if cache.exists():
            frames.append(pd.read_csv(cache))
            continue
        safe_ct = complaint_type.replace("'", "''")
        safe_desc = descriptor_like.replace("'", "''")
        params = {
            "$select": "date_trunc_ymd(created_date) AS d, date_extract_hh(created_date) AS h, borough AS b, count(*) AS n",
            "$where": (f"complaint_type='{safe_ct}' AND descriptor like '{safe_desc}' "
                       f"AND created_date >= '{lo}T00:00:00' AND created_date < '{hi}T00:00:00'"),
            "$group": "date_trunc_ymd(created_date), date_extract_hh(created_date), borough",
            "$limit": 50000,
        }
        if app_token:
            params["$$app_token"] = app_token
        log(f"  311 {rid} {lo:%Y-%m}")
        rows = _get_json(client, SOCRATA_URL.format(rid=rid), params, log)
        if len(rows) >= 50000:
            raise RuntimeError("Socrata result hit the 50,000-row limit for one month; refine the query.")
        frame = pd.DataFrame(rows, columns=["d", "h", "b", "n"]) if rows else pd.DataFrame(columns=["d", "h", "b", "n"])
        frame.to_csv(cache, index=False)
        frames.append(frame)
        time.sleep(0.3)
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["d", "h", "b", "n"])
    if all_rows.empty:
        return pd.DataFrame(columns=["local_time", "borough", "n"])
    all_rows["local_time"] = pd.to_datetime(all_rows["d"]) + pd.to_timedelta(pd.to_numeric(all_rows["h"]), unit="h")
    out = all_rows.rename(columns={"b": "borough"})[["local_time", "borough", "n"]]
    out["n"] = pd.to_numeric(out["n"])
    return out
