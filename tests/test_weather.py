import asyncio

import httpx
import pytest

from app.config import Settings
from app.services import weather_service as ws
from app.services.weather_service import LocationNotFound, WeatherService, WeatherUnavailable
from tests.helpers import forecast_payload, mock_client


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None
    monkeypatch.setattr(ws.asyncio, "sleep", instant)


def run(coro):
    return asyncio.run(coro)


def service(client, **over):
    clock = over.pop("clock", None)
    kwargs = {"clock": clock} if clock else {}
    return WeatherService(Settings(**over), client=client, **kwargs)


def test_live_fetch_then_cache_label():
    svc = service(mock_client())
    async def go():
        first = await svc.get_weather("Mumbai")
        second = await svc.get_weather("  mumbai ")
        return first, second
    first, second = run(go())
    assert first.status == "live" and first.place.name == "Mumbai" and first.snapshot.inputs.precip_past_72h_mm == 0.0
    assert second.status == "cached" and second.age_seconds >= 0


def test_unknown_location():
    svc = service(mock_client(geocode={"results": []}))
    with pytest.raises(LocationNotFound):
        run(svc.get_weather("zzzzqqq"))


def test_timeout_is_reported_not_hidden():
    svc = service(mock_client(forecast_error=httpx.ReadTimeout("slow")))
    with pytest.raises(WeatherUnavailable) as info:
        run(svc.get_weather("Mumbai"))
    assert info.value.code == "timeout" and svc.last_error


def test_rate_limit_and_provider_errors():
    with pytest.raises(WeatherUnavailable) as info:
        run(service(mock_client(forecast_status=429)).get_weather("Mumbai"))
    assert info.value.code == "rate_limited"
    with pytest.raises(WeatherUnavailable) as info:
        run(service(mock_client(forecast_status=503)).get_weather("Mumbai"))
    assert info.value.code == "provider_error"


def test_network_down():
    svc = service(mock_client(forecast_error=httpx.ConnectError("no route")))
    with pytest.raises(WeatherUnavailable) as info:
        run(svc.get_weather("Mumbai"))
    assert info.value.code == "network"


def test_malformed_payload_is_an_error_not_fake_data():
    svc = service(mock_client(forecast={"current": {}, "hourly": {}}))
    with pytest.raises(WeatherUnavailable) as info:
        run(svc.get_weather("Mumbai"))
    assert info.value.code == "bad_response"


def test_missing_fields_are_kept_as_missing():
    svc = service(mock_client(forecast=forecast_payload(humidity=None)))
    result = run(svc.get_weather("Mumbai"))
    assert result.snapshot.inputs.humidity_pct is None and "humidity_pct" in result.snapshot.missing_fields


def test_stale_data_is_labelled_stale_when_provider_fails():
    now = [1000.0]
    state = {"fail": False}
    import httpx as _h
    def handler(request):
        if "geocoding" in request.url.host:
            return _h.Response(200, json={"results": [{"name": "Mumbai", "latitude": 19.0, "longitude": 72.8, "country_code": "IN"}]})
        if state["fail"]:
            raise _h.ConnectError("down")
        return _h.Response(200, json=forecast_payload())
    client = _h.AsyncClient(transport=_h.MockTransport(handler))
    svc = WeatherService(Settings(weather_cache_ttl_s=60, weather_stale_ttl_s=3600), client=client, clock=lambda: now[0])
    async def go():
        live = await svc.get_weather("Mumbai")
        state["fail"] = True
        now[0] += 120                                  # cache expired, provider down
        stale = await svc.get_weather("Mumbai")
        now[0] += 7200                                 # too old even for stale
        try:
            await svc.get_weather("Mumbai")
            too_old = None
        except WeatherUnavailable as exc:
            too_old = exc
        return live, stale, too_old
    live, stale, too_old = run(go())
    assert live.status == "live" and stale.status == "stale" and stale.age_seconds >= 120 and "not" in stale.note.lower()
    assert too_old is not None
