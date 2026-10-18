"""Turn an Open-Meteo forecast payload into WeatherInputs.

Pure function (no network, no framework) so it can be unit-tested with fixtures.

Open-Meteo hourly ``precipitation`` at time T is the sum over the PRECEDING hour.
Therefore, with ``idx`` the index of the current hour:
  * last hour   = hourly[idx]
  * last 3 h    = hourly[idx-2 .. idx]
  * last 24 h   = hourly[idx-23 .. idx]
  * last 72 h   = hourly[idx-71 .. idx]
  * next 6 h    = hourly[idx+1 .. idx+6]
Missing values are kept as ``None``; they are never silently turned into 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .features import WeatherInputs


class ForecastFormatError(Exception):
    """The provider answered but the payload is not usable."""


@dataclass
class WeatherSnapshot:
    inputs: WeatherInputs
    observed_at: str                      # local time string from the provider
    timezone: str
    hourly_next: list[dict[str, Optional[float]]] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _window_sum(values: list, start: int, stop: int) -> Optional[float]:
    if start < 0 or stop > len(values):
        return None
    chunk = [_num(v) for v in values[start:stop]]
    if any(v is None for v in chunk):
        return None
    return float(sum(chunk))


def parse_forecast(payload: dict) -> WeatherSnapshot:
    if not isinstance(payload, dict):
        raise ForecastFormatError("Provider response was not a JSON object.")
    current = payload.get("current") or {}
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    precip = hourly.get("precipitation") or []
    probs = hourly.get("precipitation_probability") or []
    current_time = current.get("time")
    if not current_time or not times or len(precip) != len(times):
        raise ForecastFormatError("Provider response is missing the current time or hourly rainfall series.")

    hour_key = str(current_time)[:13] + ":00"
    try:
        idx = times.index(hour_key)
    except ValueError as exc:
        raise ForecastFormatError("The current hour was not found in the hourly series.") from exc

    next_precip = [_num(v) for v in precip[idx + 1: idx + 7]]
    next_probs = [_num(probs[i]) for i in range(idx + 1, min(idx + 7, len(probs)))]
    valid_probs = [p for p in next_probs if p is not None]

    inputs = WeatherInputs(
        precip_current_mm=_num(precip[idx]),
        precip_past_3h_mm=_window_sum(precip, idx - 2, idx + 1),
        precip_past_24h_mm=_window_sum(precip, idx - 23, idx + 1),
        precip_past_72h_mm=_window_sum(precip, idx - 71, idx + 1),
        precip_next_6h_mm=_window_sum(precip, idx + 1, idx + 7),
        rain_probability_pct=max(valid_probs) if valid_probs else None,
        temperature_c=_num(current.get("temperature_2m")),
        humidity_pct=_num(current.get("relative_humidity_2m")),
        wind_speed_kmh=_num(current.get("wind_speed_10m")),
    )

    hourly_next = []
    for offset in range(1, 7):
        i = idx + offset
        if i >= len(times):
            break
        hourly_next.append({
            "time": times[i],
            "precipitation_mm": _num(precip[i]),
            "probability_pct": _num(probs[i]) if i < len(probs) else None,
        })

    missing = [name for name, value in inputs.to_dict().items() if value is None]
    return WeatherSnapshot(
        inputs=inputs,
        observed_at=str(current_time),
        timezone=str(payload.get("timezone") or "local time"),
        hourly_next=hourly_next,
        missing_fields=missing,
    )
