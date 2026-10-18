import pytest
from pydantic import ValidationError

from app.schemas import LocationRequest, ManualInput, parse_manual_form


def test_location_is_trimmed_and_validated():
    assert LocationRequest(location="  Navi   Mumbai ").location == "Navi Mumbai"
    for bad in ["", " ", "a", "x" * 121]:
        with pytest.raises(ValidationError):
            LocationRequest(location=bad)
    with pytest.raises(ValidationError):
        LocationRequest(location="Mumbai", extra="nope")


def test_manual_input_ranges_and_requirements():
    assert ManualInput(precip_current_mm=3).precip_current_mm == 3
    for bad in [{"precip_current_mm": -1}, {"precip_current_mm": 1000}, {"humidity_pct": 101, "precip_current_mm": 1},
                {"temperature_c": 99, "precip_current_mm": 1}, {"precip_current_mm": float("nan")}]:
        with pytest.raises(ValidationError):
            ManualInput(**bad)
    with pytest.raises(ValidationError):
        ManualInput(temperature_c=20)                       # no rainfall value at all
    with pytest.raises(ValidationError):
        ManualInput(precip_current_mm=5, precip_past_3h_mm=2)   # nested windows must be consistent


def test_form_parsing_reports_errors_per_field():
    parsed, errors, raw = parse_manual_form({"precip_current_mm": "abc", "humidity_pct": "55"})
    assert parsed is None and "precip_current_mm" in errors and raw["humidity_pct"] == "55"
    parsed, errors, _ = parse_manual_form({"precip_current_mm": "5", "precip_past_3h_mm": "2"})
    assert parsed is None and "precip_past_3h_mm" in errors
    parsed, errors, _ = parse_manual_form({"humidity_pct": "55"})
    assert parsed is None and "__all__" in errors
    parsed, errors, _ = parse_manual_form({"precip_current_mm": "-4"})
    assert parsed is None and "Must be between" in errors["precip_current_mm"]
    parsed, errors, _ = parse_manual_form({"precip_current_mm": "2,5", "model_area": "weird"})
    assert errors == {} and parsed.precip_current_mm == 2.5 and parsed.model_area == "other"
