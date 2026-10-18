"""Feature schema shared by training, live-weather and manual prediction.

This module is deliberately free of web-framework imports so that the offline
training pipeline, the web app and the tests all use the *same* definitions.

Two different sets of inputs exist and the difference is documented on the
"How It Works" page:

* ``MODEL_FEATURES``  - the features a trained model is allowed to consume.
  They are limited to variables that exist in BOTH the historical weather
  archive used for training AND the live forecast API, so the model is never
  trained on one feature set and served another.
* ``HEURISTIC_FEATURES`` - the inputs of the transparent weather-based fallback
  (which also uses forecast rainfall; forecasts are not available historically
  in a form that matches live forecasts, so the model does not use them).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

FEATURE_SCHEMA_VERSION = "1"

# Risk classes, in increasing severity. Shared by training, serving and the UI.
RISK_CLASSES: tuple[str, ...] = ("Low", "Moderate", "High")

# Order matters: it is the column order of the numpy array passed to the model.
MODEL_FEATURES: tuple[str, ...] = (
    "precip_current_mm",
    "precip_past_3h_mm",
    "precip_past_24h_mm",
    "precip_past_72h_mm",
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
)

HEURISTIC_FEATURES: tuple[str, ...] = (
    "precip_current_mm",
    "precip_next_6h_mm",
    "rain_probability_pct",
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    unit: str
    minimum: float
    maximum: float
    step: float
    description: str


FEATURE_SPECS: dict[str, FeatureSpec] = {
    spec.name: spec
    for spec in (
        FeatureSpec("precip_current_mm", "Rain in the last hour", "mm", 0, 200, 0.1,
                    "Precipitation that fell during the hour ending now."),
        FeatureSpec("precip_past_3h_mm", "Rain in the last 3 hours", "mm", 0, 500, 0.1,
                    "Total precipitation over the 3 hours ending now (includes the last hour)."),
        FeatureSpec("precip_past_24h_mm", "Rain in the last 24 hours", "mm", 0, 1000, 0.1,
                    "Total precipitation over the 24 hours ending now (includes the last 3 hours)."),
        FeatureSpec("precip_past_72h_mm", "Rain in the last 72 hours", "mm", 0, 1500, 0.1,
                    "Total precipitation over the 72 hours ending now. A rough measure of how wet the ground and drains already are."),
        FeatureSpec("precip_next_6h_mm", "Forecast rain, next 6 hours", "mm", 0, 500, 0.1,
                    "Forecast precipitation total for the 6 hours after now. Used by the weather-based fallback only."),
        FeatureSpec("rain_probability_pct", "Peak rain probability, next 6 hours", "%", 0, 100, 1,
                    "Highest hourly precipitation probability in the next 6 hours. Used by the fallback only."),
        FeatureSpec("temperature_c", "Air temperature", "°C", -60, 60, 0.1,
                    "Air temperature at 2 m."),
        FeatureSpec("humidity_pct", "Relative humidity", "%", 0, 100, 1,
                    "Relative humidity at 2 m."),
        FeatureSpec("wind_speed_kmh", "Wind speed", "km/h", 0, 400, 0.1,
                    "Wind speed at 10 m."),
    )
}

ALL_INPUT_FIELDS: tuple[str, ...] = tuple(FEATURE_SPECS)


def _clean(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


@dataclass
class WeatherInputs:
    """All weather quantities the app can use. ``None`` means 'not available'."""

    precip_current_mm: Optional[float] = None
    precip_past_3h_mm: Optional[float] = None
    precip_past_24h_mm: Optional[float] = None
    precip_past_72h_mm: Optional[float] = None
    precip_next_6h_mm: Optional[float] = None
    rain_probability_pct: Optional[float] = None
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_kmh: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ALL_INPUT_FIELDS:
            setattr(self, name, _clean(getattr(self, name)))

    def missing(self, names: Iterable[str]) -> list[str]:
        return [n for n in names if getattr(self, n) is None]

    def range_violations(self, names: Iterable[str]) -> list[str]:
        bad = []
        for n in names:
            value = getattr(self, n)
            spec = FEATURE_SPECS[n]
            if value is not None and not (spec.minimum <= value <= spec.maximum):
                bad.append(n)
        return bad

    def model_row(self, names: Iterable[str]) -> list[float]:
        names = list(names)
        missing = self.missing(names)
        if missing:
            raise ValueError(f"missing features: {', '.join(missing)}")
        return [float(getattr(self, n)) for n in names]

    def to_dict(self) -> dict[str, Optional[float]]:
        return asdict(self)


def window_consistency_errors(values: dict[str, float]) -> dict[str, str]:
    """Rainfall windows are nested (1 h <= 3 h <= 24 h <= 72 h)."""
    chain = ("precip_current_mm", "precip_past_3h_mm", "precip_past_24h_mm", "precip_past_72h_mm")
    errors: dict[str, str] = {}
    for shorter, longer in zip(chain, chain[1:]):
        a, b = values.get(shorter), values.get(longer)
        if a is not None and b is not None and b + 1e-9 < a:
            errors[longer] = (
                f"Must be at least the '{FEATURE_SPECS[shorter].label}' value, "
                "because the longer window includes the shorter one."
            )
    return errors
