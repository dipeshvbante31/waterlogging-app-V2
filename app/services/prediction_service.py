"""One prediction service used by BOTH the live-weather and the manual-prediction paths.

Order of preference
    1. A compatible trained model, whose artefacts live in the configured models directory and
       pass integrity / schema checks, used only if every model feature is available and the
       location is inside the model's documented coverage area.
    2. A clearly labelled heuristic fallback (``weather_fallback``) in all other cases.

The method that actually produced a result is always reported, together with the *reason* a
fallback was used. A model error is logged and reported as such; it is never presented as a
model prediction.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.features import FEATURE_SCHEMA_VERSION, FEATURE_SPECS, MODEL_FEATURES, RISK_CLASSES, WeatherInputs
from app.core.geo import haversine_km
from app.core.heuristic import ADVICE, InsufficientWeatherData, weather_fallback

log = logging.getLogger("waterlogging.prediction")

MODEL_FILE = "waterlogging_model.joblib"
METADATA_FILE = "model_metadata.json"
METRICS_FILE = "model_metrics.json"
MAX_JSON_BYTES = 5_000_000

METHOD_MODEL = "trained_model"
METHOD_FALLBACK = "weather_fallback"
METHOD_LABELS = {METHOD_MODEL: "Trained ML model", METHOD_FALLBACK: "Weather-based fallback (heuristic)"}

FALLBACK_MESSAGES = {
    "model_missing": "No trained model is installed, so a transparent weather-based fallback was used.",
    "model_incompatible": "The installed model is incompatible with this version of the application, so the weather-based fallback was used.",
    "model_corrupt": "The installed model failed its integrity check or could not be loaded, so the weather-based fallback was used.",
    "model_error": "The trained model raised an error while predicting, so the weather-based fallback was used. The error was logged.",
    "outside_region": "The trained model only covers a specific area and this location is outside it, so the weather-based fallback was used.",
    "area_not_confirmed": "The readings were not marked as belonging to the area the trained model covers, so the weather-based fallback was used.",
    "missing_features": "The trained model needs every one of its inputs and some are unavailable, so the weather-based fallback was used.",
}

COMMON_LIMITATIONS = [
    "This is an informational screening estimate, not an official flood warning.",
    "Drainage capacity, terrain, tides, land cover and maintenance are not inputs.",
]


class NoPredictionPossible(Exception):
    """Neither the model nor the fallback has enough data."""


@dataclass
class ModelState:
    status: str = "missing"      # loaded | missing | incompatible | corrupt
    detail: str = ""
    version: Optional[str] = None
    name: Optional[str] = None
    trained_at: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.status == "loaded"

    def public(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail, "version": self.version, "name": self.name,
                "trained_at": self.trained_at}


def read_json_file(path: Path) -> Optional[dict]:
    """Read a small JSON file; ``None`` if absent, oversized, unreadable or not an object."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _major_minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


class PredictionService:
    def __init__(self, models_dir: Path):
        # The path comes from deployment configuration only, never from a request.
        self.models_dir = Path(models_dir)
        self.pipeline: Any = None
        self.metadata: Optional[dict] = None
        self.state = ModelState()
        self.load()

    # ------------------------------------------------------------------ loading
    def _fail(self, status: str, detail: str) -> ModelState:
        self.pipeline, self.metadata = None, None
        self.state = ModelState(status=status, detail=detail)
        level = logging.INFO if status == "missing" else logging.WARNING
        log.log(level, "model not loaded (%s): %s", status, detail)
        return self.state

    def load(self) -> ModelState:
        model_path, meta_path = self.models_dir / MODEL_FILE, self.models_dir / METADATA_FILE
        if not model_path.is_file():
            return self._fail("missing", f"{MODEL_FILE} not found. Run scripts/train_model.py to create one.")
        meta = read_json_file(meta_path)
        if meta is None:
            return self._fail("incompatible", f"{METADATA_FILE} is missing or unreadable, so the model cannot be validated.")
        if meta.get("features") != list(MODEL_FEATURES) or str(meta.get("feature_schema_version")) != FEATURE_SCHEMA_VERSION:
            return self._fail("incompatible", "The model's feature schema does not match this application. Retrain the model.")
        classes = meta.get("classes")
        if not isinstance(classes, list) or not classes or not set(classes) <= set(RISK_CLASSES):
            return self._fail("incompatible", "The model's class labels are not recognised.")
        expected_sha = meta.get("model_sha256")
        if not isinstance(expected_sha, str) or _sha256(model_path) != expected_sha:
            return self._fail("corrupt", "The model file does not match the checksum recorded at training time.")
        trained_sk = (meta.get("environment") or {}).get("scikit_learn")
        try:
            import sklearn
            if trained_sk and _major_minor(trained_sk) != _major_minor(sklearn.__version__):
                return self._fail("incompatible", f"Model was trained with scikit-learn {trained_sk} but {sklearn.__version__} "
                                                  "is installed. Retrain with this version (or install the matching one).")
            import joblib
            pipeline = joblib.load(model_path)           # trusted artefact: fixed path + checksum verified above
        except Exception as exc:
            return self._fail("corrupt", f"The model could not be loaded ({type(exc).__name__}).")
        if not hasattr(pipeline, "predict") or sorted(map(str, getattr(pipeline, "classes_", []))) != sorted(classes):
            return self._fail("incompatible", "The loaded object is not a classifier with the expected classes.")
        try:  # smoke test with a mid-range row
            probe = np.array([[spec_mid(n) for n in MODEL_FEATURES]], dtype=float)
            pipeline.predict(probe)
        except Exception as exc:
            return self._fail("incompatible", f"The model failed a smoke test ({type(exc).__name__}).")
        self.pipeline, self.metadata = pipeline, meta
        self.state = ModelState("loaded", "", meta.get("model_version"), meta.get("model_name"), meta.get("trained_at_utc"))
        log.info("trained model loaded: %s (%s)", self.state.version, self.state.name)
        return self.state

    # ------------------------------------------------------------------ coverage
    @property
    def coverage(self) -> Optional[dict]:
        return (self.metadata or {}).get("applicability") if self.state.loaded else None

    def covers(self, latitude: Optional[float], longitude: Optional[float]) -> bool:
        region = self.coverage
        if not region or latitude is None or longitude is None:
            return False
        limit = float(region.get("max_distance_km", 25))
        return any(haversine_km(latitude, longitude, float(p["latitude"]), float(p["longitude"])) <= limit
                   for p in region.get("points", []) if "latitude" in p and "longitude" in p)

    # ------------------------------------------------------------------ prediction
    def predict(self, inputs: WeatherInputs, *, latitude: Optional[float] = None, longitude: Optional[float] = None,
                area_confirmed: Optional[bool] = None, data_timestamp: Optional[str] = None) -> dict[str, Any]:
        """``area_confirmed`` is for manual input (no coordinates): the user's statement about the area."""
        reason: Optional[str] = None
        detail: Optional[str] = None
        if not self.state.loaded:
            reason = f"model_{self.state.status}"
            detail = self.state.detail
        else:
            missing = inputs.missing(MODEL_FEATURES)
            in_area = self.covers(latitude, longitude) if area_confirmed is None else bool(area_confirmed)
            if missing:
                reason = "missing_features"
                detail = "Unavailable: " + ", ".join(FEATURE_SPECS[m].label for m in missing) + "."
            elif not in_area:
                reason = "outside_region" if area_confirmed is None else "area_not_confirmed"
                region = self.coverage or {}
                detail = f"The model was trained for: {region.get('name', 'an unspecified area')}."
            else:
                try:
                    return self._model_result(inputs, data_timestamp)
                except Exception:
                    log.exception("trained model failed while predicting; using the labelled fallback")
                    reason = "model_error"
        return self._fallback_result(inputs, reason, detail, data_timestamp)

    def _model_result(self, inputs: WeatherInputs, data_timestamp: Optional[str]) -> dict[str, Any]:
        row = np.array([inputs.model_row(MODEL_FEATURES)], dtype=float)
        proba = self.pipeline.predict_proba(row)[0]
        classes = [str(c) for c in self.pipeline.classes_]
        scores = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        level = max(scores, key=scores.get)                 # argmax of the scores (what predict() returns)
        scores = {c: scores[c] for c in RISK_CLASSES if c in scores}   # stable Low/Moderate/High order for the UI
        if level not in RISK_CLASSES:
            raise ValueError(f"unexpected class {level!r}")
        importance = (self.metadata or {}).get("feature_importance") or {}
        factors = []
        for name in MODEL_FEATURES:
            spec = FEATURE_SPECS[name]
            factors.append({"name": name, "label": spec.label, "value": round(float(getattr(inputs, name)), 1), "unit": spec.unit,
                            "global_importance": (importance.get(name) or {}).get("mean_drop_in_macro_f1")})
        factors.sort(key=lambda f: (f["global_importance"] is None, -(f["global_importance"] or 0)))
        return {
            "method": METHOD_MODEL, "method_label": METHOD_LABELS[METHOD_MODEL],
            "risk_level": level, "advice": ADVICE[level],
            "model_scores": scores,
            "model_score_note": "Uncalibrated class scores from the model. They are NOT calibrated probabilities of real-world waterlogging.",
            "model_version": self.state.version, "model_name": self.state.name,
            "reasons": [f"{f['label']}: {f['value']:g} {f['unit']}" for f in factors],
            "factors": factors,
            "factor_note": "Factors are listed by the model's overall importance on its test period (permutation importance). "
                           "This is a global property, not an explanation of this single prediction.",
            "notes": [], "fallback_reason": None, "fallback_message": None,
            "data_timestamp": data_timestamp, "generated_at": _now(),
            "limitations": COMMON_LIMITATIONS + (self.metadata or {}).get("limitations", [])[:3],
        }

    def _fallback_result(self, inputs: WeatherInputs, reason: Optional[str], detail: Optional[str],
                         data_timestamp: Optional[str]) -> dict[str, Any]:
        try:
            result = weather_fallback(inputs)
        except InsufficientWeatherData as exc:
            raise NoPredictionPossible(str(exc)) from exc
        message = FALLBACK_MESSAGES.get(reason or "", "A weather-based fallback was used.")
        if detail:
            message = f"{message} {detail}"
        result.update({
            "method": METHOD_FALLBACK, "method_label": METHOD_LABELS[METHOD_FALLBACK],
            "model_version": None, "model_name": None,
            "fallback_reason": reason, "fallback_message": message,
            "data_timestamp": data_timestamp, "generated_at": _now(),
            "limitations": COMMON_LIMITATIONS + ["The fallback thresholds are hand-written and have not been validated against observed waterlogging."],
        })
        return result


def spec_mid(name: str) -> float:
    spec = FEATURE_SPECS[name]
    return (spec.minimum + min(spec.maximum, 50.0)) / 4.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
