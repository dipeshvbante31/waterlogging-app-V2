"""Pydantic request models. Ranges come from ``app.core.features.FEATURE_SPECS`` (single source of truth)."""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.features import FEATURE_SPECS, MODEL_FEATURES, WeatherInputs, window_consistency_errors

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class LocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str = Field(min_length=2, max_length=120)

    @field_validator("location")
    @classmethod
    def _clean(cls, value: str) -> str:
        value = _CONTROL.sub(" ", value).strip()
        value = re.sub(r"\s+", " ", value)
        if len(value) < 2:
            raise ValueError("Enter at least two characters for a location.")
        return value


def _field(name: str):
    spec = FEATURE_SPECS[name]
    return Field(default=None, ge=spec.minimum, le=spec.maximum, allow_inf_nan=False, description=spec.description)


class ManualInput(BaseModel):
    """Weather values typed in by the user. Every value is optional; what is missing decides which method can run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    precip_current_mm: Optional[float] = _field("precip_current_mm")
    precip_past_3h_mm: Optional[float] = _field("precip_past_3h_mm")
    precip_past_24h_mm: Optional[float] = _field("precip_past_24h_mm")
    precip_past_72h_mm: Optional[float] = _field("precip_past_72h_mm")
    precip_next_6h_mm: Optional[float] = _field("precip_next_6h_mm")
    rain_probability_pct: Optional[float] = _field("rain_probability_pct")
    temperature_c: Optional[float] = _field("temperature_c")
    humidity_pct: Optional[float] = _field("humidity_pct")
    wind_speed_kmh: Optional[float] = _field("wind_speed_kmh")
    # Does the user state that these readings are for the area the trained model was built for?
    model_area: Literal["covered", "other"] = "other"

    @model_validator(mode="after")
    def _cross_checks(self) -> "ManualInput":
        values = {n: getattr(self, n) for n in FEATURE_SPECS}
        problems = window_consistency_errors(values)
        if problems:
            field, message = next(iter(problems.items()))
            raise ValueError(f"{field}: {message}")
        if self.precip_current_mm is None and self.precip_next_6h_mm is None:
            raise ValueError("Enter current rainfall or the 6-hour forecast rainfall (or both) so a risk can be estimated.")
        return self

    def to_inputs(self) -> WeatherInputs:
        return WeatherInputs(**{n: getattr(self, n) for n in FEATURE_SPECS})


FORM_FIELDS: tuple[str, ...] = tuple(FEATURE_SPECS)
MODEL_FIELD_SET = frozenset(MODEL_FEATURES)


def parse_manual_form(form: Any) -> tuple[Optional[ManualInput], dict[str, str], dict[str, str]]:
    """Turn submitted form data into a validated ``ManualInput``.

    Returns ``(model_or_None, errors_by_field, raw_values)``. Never raises for user mistakes.
    """
    raw: dict[str, str] = {}
    data: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in FORM_FIELDS:
        text = str(form.get(name, "") or "").strip()
        raw[name] = text
        if text == "":
            continue
        try:
            data[name] = float(text.replace(",", "."))
        except ValueError:
            errors[name] = "Enter a number."
    area = str(form.get("model_area", "other") or "other")
    raw["model_area"] = area
    data["model_area"] = area if area in {"covered", "other"} else "other"
    if not errors:
        errors.update(window_consistency_errors({n: data.get(n) for n in FEATURE_SPECS}))
    if errors:
        return None, errors, raw
    try:
        return ManualInput(**data), {}, raw
    except Exception as exc:  # pydantic.ValidationError; kept broad so a form can never 500
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            for item in exc.errors():
                loc = item.get("loc") or ()
                key = str(loc[0]) if loc else "__all__"
                message = str(item.get("msg", "Invalid value")).removeprefix("Value error, ")
                if key in FEATURE_SPECS and key not in errors:
                    spec = FEATURE_SPECS[key]
                    if item.get("type") in {"greater_than_equal", "less_than_equal"}:
                        message = f"Must be between {spec.minimum:g} and {spec.maximum:g} {spec.unit}."
                    errors[key] = message
                elif key == "__all__" or key not in FEATURE_SPECS:
                    errors.setdefault("__all__", message)
            return None, errors or {"__all__": "Invalid input."}, raw
        raise
