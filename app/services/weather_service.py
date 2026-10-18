"""Open-Meteo geocoding + forecast client with timeouts, a TTL cache and honest freshness labels.

* ``live``   - fetched from the provider during this request.
* ``cached`` - served from the in-memory cache; younger than ``weather_cache_ttl_s``.
* ``stale``  - the provider failed and an older cached copy (younger than ``weather_stale_ttl_s``) is
               being served. It is labelled as such and must never be shown as live.
If no usable data exists a ``WeatherError`` is raised; nothing is ever invented.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from app.config import Settings
from app.core.forecast_parser import ForecastFormatError, WeatherSnapshot, parse_forecast

log = logging.getLogger("waterlogging.weather")

CURRENT_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
HOURLY_VARS = "precipitation,precipitation_probability"
MAX_CACHE_ENTRIES = 256


class WeatherError(Exception):
    code = "weather_error"
    http_status = 503

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class LocationNotFound(WeatherError):
    code = "location_not_found"
    http_status = 404


class WeatherUnavailable(WeatherError):
    code = "weather_unavailable"
    http_status = 503


@dataclass(frozen=True)
class Place:
    name: str
    admin1: Optional[str]
    country: Optional[str]
    latitude: float
    longitude: float


@dataclass
class WeatherResult:
    place: Place
    snapshot: WeatherSnapshot
    status: str                  # "live" | "cached" | "stale"
    fetched_at: datetime         # when the provider was actually queried (UTC)
    age_seconds: int
    note: Optional[str] = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class WeatherService:
    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None,
                 clock: Callable[[], float] = time.time):
        self._settings = settings
        self._client = client
        self._owns_client = client is None
        self._clock = clock
        self._cache: dict[str, tuple[float, Place, WeatherSnapshot]] = {}
        self.last_success_at: Optional[datetime] = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ plumbing
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(self._settings.http_timeout_s, connect=min(5.0, self._settings.http_timeout_s))
            self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=False,
                                             headers={"User-Agent": "WaterloggingWatch/2.0 (educational project)"})
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        """GET with one retry for transient transport errors / 502-504. 4xx and 429 are not retried."""
        last: Optional[WeatherError] = None
        for attempt in range(2):
            try:
                response = await self._http().get(url, params=params)
            except httpx.TimeoutException:
                last = WeatherUnavailable("The weather service took too long to respond.", code="timeout")
            except httpx.TransportError:
                last = WeatherUnavailable("Could not reach the weather service. Check the network connection.", code="network")
            else:
                status = response.status_code
                if status == 429:
                    raise WeatherUnavailable("The weather service is rate-limiting requests. Please try again in a minute.",
                                             code="rate_limited")
                if status in (502, 503, 504):
                    last = WeatherUnavailable("The weather service is temporarily unavailable.", code="provider_error")
                elif status >= 400:
                    raise WeatherUnavailable("The weather service rejected the request.", code="provider_error")
                else:
                    try:
                        return response.json()
                    except ValueError:
                        raise WeatherUnavailable("The weather service returned an unreadable response.", code="bad_response")
            if attempt == 0:
                await asyncio.sleep(0.4)
        assert last is not None
        raise last

    # ------------------------------------------------------------------ provider calls
    async def geocode(self, query: str) -> Place:
        data = await self._get_json(self._settings.geocoding_url,
                                    {"name": query, "count": 5, "language": "en", "format": "json"})
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            raise LocationNotFound("We couldn't find that place. Try a city or a more specific area name.")
        valid = [r for r in results if isinstance(r, dict) and _is_number(r.get("latitude")) and _is_number(r.get("longitude"))]
        if not valid:
            raise LocationNotFound("We couldn't find usable coordinates for that place.")
        # The app was built for India: prefer an Indian match when several are returned, otherwise the first.
        chosen = next((r for r in valid if r.get("country_code") == "IN"), valid[0])
        return Place(name=str(chosen.get("name") or query), admin1=chosen.get("admin1"), country=chosen.get("country"),
                     latitude=float(chosen["latitude"]), longitude=float(chosen["longitude"]))

    async def _fetch_forecast(self, place: Place) -> WeatherSnapshot:
        params = {
            "latitude": place.latitude, "longitude": place.longitude,
            "current": CURRENT_VARS, "hourly": HOURLY_VARS,
            "past_days": 3, "forecast_days": 2, "timezone": "auto",
        }
        payload = await self._get_json(self._settings.forecast_url, params)
        try:
            return parse_forecast(payload)
        except ForecastFormatError as exc:
            raise WeatherUnavailable(f"The weather service returned incomplete data ({exc}).", code="bad_response") from exc

    # ------------------------------------------------------------------ public API
    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.lower().split())

    def _remember(self, key: str, place: Place, snapshot: WeatherSnapshot) -> None:
        if len(self._cache) >= MAX_CACHE_ENTRIES:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            self._cache.pop(oldest, None)
        self._cache[key] = (self._clock(), place, snapshot)

    async def get_weather(self, query: str) -> WeatherResult:
        key = self._key(query)
        now = self._clock()
        entry = self._cache.get(key)
        if entry and now - entry[0] < self._settings.weather_cache_ttl_s:
            stamp, place, snapshot = entry
            return WeatherResult(place, snapshot, "cached", datetime.fromtimestamp(stamp, timezone.utc), int(now - stamp),
                                 "Served from a short-lived cache; the provider was not queried for this request.")
        try:
            place = await self.geocode(query)
            snapshot = await self._fetch_forecast(place)
        except LocationNotFound:
            raise                                    # a wrong place name is not a provider outage
        except WeatherError as exc:
            self.last_error = f"{exc.code}: {exc.message}"
            log.warning("weather fetch failed (%s) for query_len=%d", exc.code, len(query))
            if entry and now - entry[0] < self._settings.weather_stale_ttl_s:
                stamp, place, snapshot = entry
                return WeatherResult(place, snapshot, "stale", datetime.fromtimestamp(stamp, timezone.utc), int(now - stamp),
                                     "Live weather could not be retrieved, so older data is shown. Do not treat it as current.")
            raise
        self._remember(key, place, snapshot)
        self.last_success_at = datetime.now(timezone.utc)
        self.last_error = None
        return WeatherResult(place, snapshot, "live", self.last_success_at, 0)
