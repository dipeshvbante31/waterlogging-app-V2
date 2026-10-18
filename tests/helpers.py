"""Builders for Open-Meteo-shaped payloads and mock transports. No network is ever used."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Optional

CURRENT_TIME = "2026-10-17T14:15"
CURRENT_IDX = 3 * 24 + 14          # index of 2026-10-17T14:00 in the series below


def forecast_payload(rain: Callable[[int], Optional[float]] = lambda i: 0.0, *, current_time: str = CURRENT_TIME,
                     humidity: Optional[float] = 70.0, prob: Callable[[int], Optional[float]] = lambda i: 20.0) -> dict:
    """3 past days + 2 forecast days of hourly data starting 2026-10-14 00:00; ``i`` is the hour index."""
    start = datetime(2026, 10, 14)
    times = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(120)]
    return {
        "timezone": "Asia/Kolkata",
        "current": {"time": current_time, "temperature_2m": 29.5, "relative_humidity_2m": humidity,
                    "precipitation": rain(CURRENT_IDX), "wind_speed_10m": 12.3},
        "hourly": {"time": times, "precipitation": [rain(i) for i in range(120)],
                   "precipitation_probability": [prob(i) for i in range(120)]},
    }


GEOCODE_OK = {"results": [{"name": "Mumbai", "admin1": "Maharashtra", "country": "India", "country_code": "IN",
                           "latitude": 19.07283, "longitude": 72.88261}]}


def mock_client(forecast: dict | Callable[[], dict] | None = None, geocode: dict | None = None, *,
                forecast_error: Optional[Exception] = None, forecast_status: int = 200) -> "httpx.AsyncClient":
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding" in request.url.host:
            return httpx.Response(200, json=geocode if geocode is not None else GEOCODE_OK)
        if forecast_error is not None:
            raise forecast_error
        if forecast_status != 200:
            return httpx.Response(forecast_status, json={"reason": "mock"})
        body = forecast() if callable(forecast) else (forecast if forecast is not None else forecast_payload())
        return httpx.Response(200, json=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))
