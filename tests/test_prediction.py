import json
import shutil

import pytest

from app.core.features import MODEL_FEATURES, WeatherInputs
from app.services.prediction_service import FALLBACK_MESSAGES, NoPredictionPossible, PredictionService
from tests.conftest import INSIDE, OUTSIDE

FULL = dict(precip_current_mm=4.0, precip_past_3h_mm=9.0, precip_past_24h_mm=20.0, precip_past_72h_mm=35.0,
            precip_next_6h_mm=12.0, rain_probability_pct=80, temperature_c=24.0, humidity_pct=90, wind_speed_kmh=15.0)


def inputs(**over):
    return WeatherInputs(**{**FULL, **over})


@pytest.fixture()
def models_copy(trained_models_dir, tmp_path):
    dest = tmp_path / "models"
    shutil.copytree(trained_models_dir, dest)
    return dest


def test_missing_model_gives_labelled_fallback(tmp_path):
    svc = PredictionService(tmp_path / "nothing")
    assert svc.state.status == "missing" and not svc.state.loaded
    r = svc.predict(inputs(), latitude=INSIDE[0], longitude=INSIDE[1])
    assert r["method"] == "weather_fallback" and "heuristic" in r["method_label"].lower()
    assert r["fallback_reason"] == "model_missing" and r["model_version"] is None and "model_scores" not in r
    assert "not a probability" in r["risk_index_label"]


def test_model_in_region_with_all_features_is_used(trained_models_dir):
    svc = PredictionService(trained_models_dir)
    assert svc.state.status == "loaded", svc.state.detail
    r = svc.predict(inputs(), latitude=INSIDE[0], longitude=INSIDE[1], data_timestamp="2026-10-17T14:15")
    assert r["method"] == "trained_model" and r["method_label"] == "Trained ML model"
    assert r["risk_level"] in ("Low", "Moderate", "High") and r["fallback_reason"] is None
    assert r["model_version"] == svc.state.version
    assert abs(sum(r["model_scores"].values()) - 1.0) < 1e-3 and list(r["model_scores"]) == ["Low", "Moderate", "High"][: len(r["model_scores"])]
    assert "NOT calibrated" in r["model_score_note"] and "risk_index" not in r
    assert r["data_timestamp"] == "2026-10-17T14:15"
    assert {f["name"] for f in r["factors"]} == set(MODEL_FEATURES)


def test_outside_region_uses_fallback_with_reason(trained_models_dir):
    r = PredictionService(trained_models_dir).predict(inputs(), latitude=OUTSIDE[0], longitude=OUTSIDE[1])
    assert r["method"] == "weather_fallback" and r["fallback_reason"] == "outside_region"
    assert "SYNTHETIC test region" in r["fallback_message"]


def test_unknown_location_is_not_assumed_to_be_covered(trained_models_dir):
    r = PredictionService(trained_models_dir).predict(inputs())
    assert r["fallback_reason"] == "outside_region"


def test_missing_model_feature_falls_back_and_names_it(trained_models_dir):
    r = PredictionService(trained_models_dir).predict(inputs(precip_past_72h_mm=None), latitude=INSIDE[0], longitude=INSIDE[1])
    assert r["method"] == "weather_fallback" and r["fallback_reason"] == "missing_features"
    assert "72 hours" in r["fallback_message"]


def test_manual_area_confirmation(trained_models_dir):
    svc = PredictionService(trained_models_dir)
    assert svc.predict(inputs(), area_confirmed=True)["method"] == "trained_model"
    r = svc.predict(inputs(), area_confirmed=False)
    assert r["method"] == "weather_fallback" and r["fallback_reason"] == "area_not_confirmed"


def test_model_error_is_reported_not_disguised(trained_models_dir):
    svc = PredictionService(trained_models_dir)

    class Boom:
        classes_ = svc.pipeline.classes_
        def predict(self, X): raise RuntimeError("boom")
        def predict_proba(self, X): raise RuntimeError("boom")

    svc.pipeline = Boom()
    r = svc.predict(inputs(), latitude=INSIDE[0], longitude=INSIDE[1])
    assert r["method"] == "weather_fallback" and r["fallback_reason"] == "model_error"
    assert r["fallback_message"] == FALLBACK_MESSAGES["model_error"]


def test_corrupted_model_file_is_rejected(models_copy):
    path = models_copy / "waterlogging_model.joblib"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    svc = PredictionService(models_copy)
    assert svc.state.status == "corrupt"
    assert svc.predict(inputs(), latitude=INSIDE[0], longitude=INSIDE[1])["fallback_reason"] == "model_corrupt"


def test_garbage_model_with_matching_checksum_is_rejected(models_copy):
    import hashlib
    path = models_copy / "waterlogging_model.joblib"
    path.write_bytes(b"this is not a joblib file")
    meta_path = models_copy / "model_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta))
    assert PredictionService(models_copy).state.status == "corrupt"


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(features=list(reversed(m["features"]))),
    lambda m: m.update(feature_schema_version="999"),
    lambda m: m.update(classes=["Safe", "Danger"]),
    lambda m: m["environment"].update(scikit_learn="0.20.0"),
])
def test_incompatible_metadata_is_rejected(models_copy, mutate):
    meta_path = models_copy / "model_metadata.json"
    meta = json.loads(meta_path.read_text())
    mutate(meta)
    meta_path.write_text(json.dumps(meta))
    svc = PredictionService(models_copy)
    assert svc.state.status == "incompatible"
    assert svc.predict(inputs())["fallback_reason"] == "model_incompatible"


def test_unreadable_or_missing_metadata_is_rejected(models_copy):
    (models_copy / "model_metadata.json").write_text("{not json")
    assert PredictionService(models_copy).state.status == "incompatible"
    (models_copy / "model_metadata.json").unlink()
    assert PredictionService(models_copy).state.status == "incompatible"


def test_only_the_fixed_filename_is_ever_loaded(tmp_path, trained_models_dir):
    other = tmp_path / "m"
    other.mkdir()
    shutil.copy(trained_models_dir / "waterlogging_model.joblib", other / "evil.joblib")
    assert PredictionService(other).state.status == "missing"


def test_no_rainfall_data_means_no_prediction(tmp_path):
    with pytest.raises(NoPredictionPossible):
        PredictionService(tmp_path / "x").predict(WeatherInputs(temperature_c=20.0))
