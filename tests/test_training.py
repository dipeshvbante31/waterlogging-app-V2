import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app.core.features import FEATURE_SCHEMA_VERSION, MODEL_FEATURES
from app.ml.dataset import forward_window_counts, build_features
from app.ml.training import (TrainConfig, TrainingAborted, chronological_split, clean_dataset, label_counts,
                             run_training, thresholds_from_train)
from tests.synthetic import synthetic_dataset


def test_forward_window_counts_only_look_forward():
    idx = pd.date_range("2024-01-01", periods=30, freq="h")
    counts = pd.Series(0, index=idx)
    counts.iloc[10] = 1
    out = forward_window_counts(counts, idx, window=6)
    # hours 5..10 can "see" the event at hour 10; hour 11 onwards and hour 4 cannot.
    assert [int(out.iloc[i]) for i in (4, 5, 10, 11)] == [0, 1, 1, 0]
    assert out.iloc[-5:].isna().all()                      # windows running off the end are unknown


def test_features_use_only_past_and_present_rain():
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    rain = pd.Series(0.0, index=idx)
    rain.iloc[50] = 7.0
    weather = pd.DataFrame({"precipitation": rain, "temperature_2m": 20.0, "relative_humidity_2m": 60.0, "wind_speed_10m": 5.0})
    f = build_features(weather)
    assert f["precip_past_3h_mm"].iloc[49] == 0.0           # rain at hour 50 is the future for hour 49
    assert f["precip_past_3h_mm"].iloc[50] == 7.0 and f["precip_past_3h_mm"].iloc[52] == 7.0
    assert f["precip_past_3h_mm"].iloc[53] == 0.0


def test_chronological_split_has_purge_gap():
    df = synthetic_dataset(n_units=1, start="2022-01-01", end="2022-06-30")
    train, test = chronological_split(df, "2022-05-01", purge_hours=6)
    assert train["timestamp_utc"].max() < pd.Timestamp("2022-05-01") - pd.Timedelta(hours=6) + pd.Timedelta(seconds=1)
    assert test["timestamp_utc"].min() >= pd.Timestamp("2022-05-01")


def test_thresholds_come_from_training_counts_only():
    t = thresholds_from_train(pd.Series([0, 0, 1, 1, 2, 3, 10]), 0.9)
    assert t["moderate_min"] == 1 and t["high_min"] >= 2
    labels = label_counts([0, 1, t["high_min"]], t)
    assert list(labels) == ["Low", "Moderate", "High"]


def test_clean_dataset_drops_bad_rows():
    df = synthetic_dataset(n_units=1, start="2022-01-01", end="2022-02-15")
    bad = df.iloc[:2].copy()
    bad["unit_id"] = "bad_unit"                                   # distinct keys, so they are not removed as duplicates
    bad.iloc[0, bad.columns.get_loc("humidity_pct")] = 400.0       # out of range
    bad.iloc[1, bad.columns.get_loc("precip_current_mm")] = 5.0    # 1 h rain larger than the 3 h total: impossible
    bad.iloc[1, bad.columns.get_loc("precip_past_3h_mm")] = 1.0
    duplicate = df.iloc[[10]].copy()
    cleaned, report = clean_dataset(pd.concat([df, bad, duplicate], ignore_index=True))
    assert report["dropped_duplicate_unit_hour"] == 1
    assert report["dropped_out_of_range"] == 1
    assert report["dropped_inconsistent_rain_windows"] == 1
    assert "bad_unit" not in set(cleaned["unit_id"]) and cleaned["humidity_pct"].max() <= 100


def test_training_aborts_without_events_and_saves_nothing(tmp_path):
    df = synthetic_dataset(n_units=1, start="2022-01-01", end="2022-03-31")
    df["window_event_count"] = 0
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(TrainingAborted):
        run_training(TrainConfig(dataset_path=csv, models_dir=tmp_path / "m", test_start="2022-03-01", log=lambda m: None))
    assert not (tmp_path / "m" / "waterlogging_model.joblib").exists()


def test_training_aborts_on_missing_columns(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    with pytest.raises(TrainingAborted):
        run_training(TrainConfig(dataset_path=csv, models_dir=tmp_path / "m", log=lambda m: None))


def test_training_aborts_when_a_class_is_too_rare(tmp_path):
    df = synthetic_dataset(n_units=3, start="2021-01-01", end="2021-12-31")
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(TrainingAborted, match="fewer than"):
        run_training(TrainConfig(dataset_path=csv, models_dir=tmp_path / "m", test_start="2021-10-01",
                                 min_class_count=10_000_000, log=lambda m: None))


def test_end_to_end_artefacts_on_synthetic_data(trained_models_dir):
    meta = json.loads((trained_models_dir / "model_metadata.json").read_text())
    metrics = json.loads((trained_models_dir / "model_metrics.json").read_text())
    model_file = trained_models_dir / "waterlogging_model.joblib"
    assert meta["features"] == list(MODEL_FEATURES) and meta["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert meta["model_sha256"] == hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert meta["scores_are_calibrated_probabilities"] is False
    assert meta["split"]["strategy"] == "chronological hold-out" and meta["model_name"] in meta["selection"]["candidates"]
    assert meta["applicability"]["name"] == "SYNTHETIC test region"
    t = metrics["test"]
    assert np.asarray(t["confusion_matrix"]).sum() == t["n"] == meta["dataset"]["counts"]["test_records"]
    assert metrics["model_version"] == meta["model_version"]
    assert {"majority_class", "rainfall_rule"} <= set(metrics["baselines"])
    assert pd.Timestamp(metrics["time_range"]["test_first_utc"]) >= pd.Timestamp("2022-07-01")
    assert pd.Timestamp(metrics["time_range"]["train_last_utc"]) < pd.Timestamp("2022-07-01")
