import math

import pytest

from app.core.features import WeatherInputs, window_consistency_errors
from app.core.forecast_parser import ForecastFormatError, parse_forecast
from app.core.heuristic import InsufficientWeatherData, weather_fallback
from tests.helpers import CURRENT_IDX, forecast_payload


def test_parser_computes_trailing_and_forward_windows():
    # 1 mm in every hour: windows are simply their lengths.
    snap = parse_forecast(forecast_payload(rain=lambda i: 1.0))
    i = snap.inputs
    assert i.precip_current_mm == 1.0
    assert i.precip_past_3h_mm == 3.0
    assert i.precip_past_24h_mm == 24.0
    assert i.precip_past_72h_mm == 72.0
    assert i.precip_next_6h_mm == 6.0
    assert len(snap.hourly_next) == 6
    assert snap.missing_fields == []


def test_parser_next_hours_exclude_current_hour():
    # put 10 mm only in the hour after the current one (15:00).
    snap = parse_forecast(forecast_payload(rain=lambda i: 10.0 if i == CURRENT_IDX + 1 else 0.0))
    assert snap.inputs.precip_next_6h_mm == 10.0
    assert snap.inputs.precip_current_mm == 0.0


def test_missing_values_stay_missing_not_zero():
    snap = parse_forecast(forecast_payload(rain=lambda i: None if i == CURRENT_IDX - 1 else 0.0, humidity=None))
    assert snap.inputs.precip_past_3h_mm is None          # a gap inside the window -> unknown, not 0
    assert snap.inputs.humidity_pct is None
    assert "humidity_pct" in snap.missing_fields and "precip_past_3h_mm" in snap.missing_fields


def test_parser_rejects_unusable_payloads():
    with pytest.raises(ForecastFormatError):
        parse_forecast({"current": {}, "hourly": {}})
    with pytest.raises(ForecastFormatError):
        parse_forecast("not a dict")                      # type: ignore[arg-type]
    bad = forecast_payload()
    bad["current"]["time"] = "2030-01-01T00:15"           # hour not present in the series
    with pytest.raises(ForecastFormatError):
        parse_forecast(bad)


def test_nan_and_inf_are_treated_as_missing():
    inputs = WeatherInputs(precip_current_mm=math.nan, temperature_c=math.inf, humidity_pct=50)
    assert inputs.precip_current_mm is None and inputs.temperature_c is None and inputs.humidity_pct == 50


def test_window_consistency():
    assert window_consistency_errors({"precip_current_mm": 5, "precip_past_3h_mm": 2}) != {}
    assert window_consistency_errors({"precip_current_mm": 2, "precip_past_3h_mm": 5, "precip_past_24h_mm": 9}) == {}


@pytest.mark.parametrize("current,nxt,expected", [(0, 0, "Low"), (4, 0, "Moderate"), (0, 20, "Moderate"), (16, 0, "High"), (0, 60, "High")])
def test_heuristic_levels(current, nxt, expected):
    result = weather_fallback(WeatherInputs(precip_current_mm=current, precip_next_6h_mm=nxt))
    assert result["risk_level"] == expected
    assert "not a probability" in result["risk_index_label"]
    assert 0 <= result["risk_index"] <= 100


def test_heuristic_requires_some_rain_data_and_reports_gaps():
    with pytest.raises(InsufficientWeatherData):
        weather_fallback(WeatherInputs(temperature_c=20))
    result = weather_fallback(WeatherInputs(precip_current_mm=1.0))
    assert any("Forecast rainfall was unavailable" in n for n in result["notes"])
