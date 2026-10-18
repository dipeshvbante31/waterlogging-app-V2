"""Build the supervised training table from hourly weather + dated incident counts.

Definitions (also written to the model metadata):

* One sample = one *unit* (a location with its own weather point) at one UTC hour ``t``.
* Features use only information up to and including hour ``t``:
  rainfall in the last hour / 3 h / 24 h / 72 h, plus temperature, humidity, wind at ``t``.
* ``window_event_count`` = number of observed incident reports created in the
  6 hours starting at ``t`` (hour buckets ``t .. t+5``). Events before ``t`` are
  never counted, so labels are strictly in the future of the features.
* The class label (Low / Moderate / High) is derived from ``window_event_count``
  at *training* time, with thresholds estimated on the training period only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_HOURS = 6
WEATHER_COLUMNS = ("precipitation", "temperature_2m", "relative_humidity_2m", "wind_speed_10m")


def weather_frame(times: list, columns: dict[str, list]) -> pd.DataFrame:
    """Hourly UTC weather -> DataFrame indexed by tz-naive UTC timestamps."""
    index = pd.to_datetime(pd.Series(times), utc=True).dt.tz_localize(None)
    frame = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce") for k, v in columns.items()})
    frame.index = pd.DatetimeIndex(index)
    return frame[~frame.index.duplicated(keep="first")].sort_index()


def build_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Rolling rainfall features. Windows are *trailing and inclusive* of hour t."""
    missing = [c for c in WEATHER_COLUMNS if c not in weather.columns]
    if missing:
        raise ValueError(f"weather frame lacks columns: {missing}")
    full = pd.date_range(weather.index.min(), weather.index.max(), freq="h")
    w = weather.reindex(full)
    rain = w["precipitation"]
    out = pd.DataFrame(index=full)
    out["precip_current_mm"] = rain
    out["precip_past_3h_mm"] = rain.rolling(3, min_periods=3).sum()
    out["precip_past_24h_mm"] = rain.rolling(24, min_periods=24).sum()
    out["precip_past_72h_mm"] = rain.rolling(72, min_periods=72).sum()
    out["temperature_c"] = w["temperature_2m"]
    out["humidity_pct"] = w["relative_humidity_2m"]
    out["wind_speed_kmh"] = w["wind_speed_10m"]
    return out


def forward_window_counts(counts: pd.Series, index: pd.DatetimeIndex, window: int = WINDOW_HOURS) -> pd.Series:
    """For each hour t: sum of ``counts`` over buckets t .. t+window-1 (NaN if window runs off the end)."""
    c = counts.groupby(level=0).sum().reindex(index, fill_value=0).astype(float)
    reversed_roll = c.iloc[::-1].rolling(window, min_periods=window).sum()
    return reversed_roll.iloc[::-1]


def local_to_utc_hours(timestamps: pd.Series, tz: str) -> pd.Series:
    """Local wall-clock timestamps -> tz-naive UTC. DST-ambiguous / non-existent times become NaT."""
    ts = pd.to_datetime(timestamps)
    aware = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    return aware.dt.tz_convert("UTC").dt.tz_localize(None)


def assemble_unit(unit_id: str, latitude: float, longitude: float,
                  weather: pd.DataFrame, counts: pd.Series) -> tuple[pd.DataFrame, dict]:
    feats = build_features(weather)
    feats["window_event_count"] = forward_window_counts(counts, feats.index)
    before = len(feats)
    feats = feats.dropna()
    feats.insert(0, "timestamp_utc", feats.index)
    feats.insert(1, "unit_id", unit_id)
    feats.insert(2, "latitude", float(latitude))
    feats.insert(3, "longitude", float(longitude))
    feats["window_event_count"] = feats["window_event_count"].round().astype(int)
    report = {"unit_id": unit_id, "hours_total": before, "hours_dropped_incomplete": before - len(feats),
              "hours_kept": len(feats), "events_in_kept_hours": int(feats["window_event_count"].sum())}
    return feats.reset_index(drop=True), report


def assemble_dataset(units: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """units: [{unit_id, latitude, longitude, weather, counts}]"""
    frames, reports = [], []
    for u in units:
        df, rep = assemble_unit(u["unit_id"], u["latitude"], u["longitude"], u["weather"], u["counts"])
        frames.append(df)
        reports.append(rep)
    if not frames:
        raise ValueError("no units supplied")
    return pd.concat(frames, ignore_index=True), reports
