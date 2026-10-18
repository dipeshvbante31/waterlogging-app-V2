"""Transparent weather-based fallback estimate.

This is a hand-written rule, NOT a trained model and NOT a probability. The
thresholds below are unchanged from the original Waterlogging Watch app. They
have not been validated against observed waterlogging.
"""
from __future__ import annotations

from typing import Any

from .features import FEATURE_SPECS, WeatherInputs

RISK_INDEX_LABEL = "Weather risk index (0-100; not a probability)"

ADVICE = {
    "High": "Heavy rain can overwhelm local drainage. Avoid flooded roads and follow local authority advisories.",
    "Moderate": "Rain may cause local water accumulation, especially in low-lying areas. Check local updates before travelling.",
    "Low": "Weather indicators suggest lower immediate rain-related risk, but local drainage and street conditions can differ.",
}


class InsufficientWeatherData(Exception):
    """Raised when neither current nor forecast rainfall is available."""


def weather_fallback(inputs: WeatherInputs) -> dict[str, Any]:
    if inputs.precip_current_mm is None and inputs.precip_next_6h_mm is None:
        raise InsufficientWeatherData("Neither current nor forecast rainfall is available.")

    notes: list[str] = []
    current = max(0.0, inputs.precip_current_mm) if inputs.precip_current_mm is not None else 0.0
    nxt = max(0.0, inputs.precip_next_6h_mm) if inputs.precip_next_6h_mm is not None else 0.0
    prob = min(100.0, max(0.0, inputs.rain_probability_pct)) if inputs.rain_probability_pct is not None else 0.0
    if inputs.precip_current_mm is None:
        notes.append("Current rainfall was unavailable and contributed nothing to the index.")
    if inputs.precip_next_6h_mm is None:
        notes.append("Forecast rainfall was unavailable and contributed nothing to the index.")
    if inputs.rain_probability_pct is None:
        notes.append("Rain probability was unavailable and contributed nothing to the index.")

    parts = {
        "precip_current_mm": current * 5.0,
        "precip_next_6h_mm": nxt * 2.2,
        "rain_probability_pct": prob * 0.25,
    }
    score = min(100.0, sum(parts.values()))

    if current >= 15 or nxt >= 50 or score >= 65:
        level = "High"
    elif current >= 3 or nxt >= 15 or score >= 30:
        level = "Moderate"
    else:
        level = "Low"

    values = {"precip_current_mm": current, "precip_next_6h_mm": nxt, "rain_probability_pct": prob}
    factors = []
    for name, contribution in parts.items():
        spec = FEATURE_SPECS[name]
        factors.append({
            "name": name,
            "label": spec.label,
            "value": round(values[name], 1),
            "unit": spec.unit,
            "contribution": round(contribution, 1),
        })
    reasons = [f"{f['label']}: {f['value']:g} {f['unit']} (adds {f['contribution']:g} to the index)" for f in factors]
    return {
        "risk_level": level,
        "risk_index": round(score),
        "risk_index_label": RISK_INDEX_LABEL,
        "advice": ADVICE[level],
        "reasons": reasons,
        "factors": factors,
        "notes": notes,
    }
