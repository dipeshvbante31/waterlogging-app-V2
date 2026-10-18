"""Safe loading of the trained model and its metadata / metrics.

* The model directory is fixed by configuration (never by request data).
* Before unpickling, the joblib file's SHA-256 is compared with the hash recorded at training time.
* Schema, feature order and scikit-learn version are checked, and a smoke prediction is run.
* Every failure ends in a clearly named status; the app then uses its labelled fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.core.features import FEATURE_SCHEMA_VERSION, MODEL_FEATURES
from app.core.geo import haversine_km

log = logging.getLogger("waterlogging.model")

MODEL_FILE = "waterlogging_model.joblib"
METADATA_FILE = "model_metadata.json"
METRICS_FILE = "model_metrics.json"

STATUS_TEXT = {
    "ready": "A trained model is loaded.",
    "not_trained": "No trained model has been produced yet. Run the training pipeline (see the README).",
    "corrupt_metadata": "The model metadata file could not be read.",
    "integrity_failed": "The model file does not match the hash recorded when it was trained, so it was not loaded.",
    "incompatible": "The saved model is not compatible with this version of the app.",
    "load_error": "The model file could not be loaded.",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _major_minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


class ModelRegistry:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model: Any = None
        self.metadata: Optional[dict] = None
        self.metrics: Optional[dict] = None
        self.status = "not_trained"
        self.detail = STATUS_TEXT["not_trained"]

    # ------------------------------------------------------------------ loading
    def _fail(self, status: str, detail: str = "") -> None:
        self.model = None
        self.status = status
        self.detail = f"{STATUS_TEXT[status]} {detail}".strip()
        log.warning("model not usable: status=%s %s", status, detail)

    def load(self) -> None:
        self.model, self.metadata, self.metrics = None, None, None
        model_path = self.model_dir / MODEL_FILE
        meta_path = self.model_dir / METADATA_FILE
        metrics_path = self.model_dir / METRICS_FILE

        if not model_path.exists():
            self.status, self.detail = "not_trained", STATUS_TEXT["not_trained"]
            log.info("no trained model found in %s; using fallback", self.model_dir)
            return
        try:
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(self.metadata, dict):
                raise ValueError("metadata is not a JSON object")
        except (OSError, ValueError) as exc:
            self.metadata = None
            return self._fail("corrupt_metadata", str(exc))
        if metrics_path.exists():
            try:
                loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.metrics = loaded if isinstance(loaded, dict) else None
            except (OSError, ValueError):
                log.warning("metrics file unreadable; performance page will say so")

        try:
            if _sha256(model_path) != self.metadata.get("model_sha256"):
                return self._fail("integrity_failed")
        except OSError as exc:
            return self._fail("load_error", str(exc))

        if self.metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            return self._fail("incompatible", "Feature schema version differs.")
        if list(self.metadata.get("features") or []) != list(MODEL_FEATURES):
            return self._fail("incompatible", "Feature list or order differs from the app's schema.")
        try:
            import sklearn
            trained_with = (self.metadata.get("environment") or {}).get("scikit_learn")
            if trained_with and _major_minor(trained_with) != _major_minor(sklearn.__version__):
                return self._fail("incompatible", f"Trained with scikit-learn {trained_with}, running {sklearn.__version__}.")
            import joblib
            model = joblib.load(model_path)
            import numpy as np
            probe = model.predict(np.zeros((1, len(MODEL_FEATURES))))
            classes = [str(c) for c in model.classes_]
            if not set(classes) <= {"Low", "Moderate", "High"} or str(probe[0]) not in classes:
                return self._fail("incompatible", "Unexpected class labels.")
        except Exception as exc:  # noqa: BLE001 - any failure here must degrade to fallback
            return self._fail("load_error", f"{type(exc).__name__}: {exc}")

        self.model = model
        self.status, self.detail = "ready", STATUS_TEXT["ready"]
        log.info("loaded model %s (%s)", self.version, self.metadata.get("model_name"))

    # --------------------------------------------------------------- properties
    @property
    def ready(self) -> bool:
        return self.model is not None and self.status == "ready"

    @property
    def version(self) -> Optional[str]:
        return (self.metadata or {}).get("model_version")

    @property
    def region_name(self) -> Optional[str]:
        return ((self.metadata or {}).get("applicability") or {}).get("name")

    def covers(self, lat: float, lon: float) -> bool:
        app = (self.metadata or {}).get("applicability") or {}
        points = app.get("points") or []
        limit = float(app.get("max_distance_km") or 0)
        if not points or limit <= 0:
            return False
        return min(haversine_km(lat, lon, float(p["latitude"]), float(p["longitude"])) for p in points) <= limit

    def public_status(self) -> dict:
        md = self.metadata or {}
        return {
            "status": self.status, "detail": self.detail, "ready": self.ready,
            "model_version": self.version if self.ready else None,
            "model_name": md.get("model_name") if self.ready else None,
            "trained_at_utc": md.get("trained_at_utc") if self.ready else None,
            "region": self.region_name if self.ready else None,
            "metrics_available": bool(self.metrics) and self.ready,
        }
