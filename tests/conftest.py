"""Shared fixtures.

IMPORTANT: ``trained_models_dir`` trains on SYNTHETIC data inside a temporary directory. It exists only to
exercise code paths (loading, schema checks, labelling). It is never written to ``models/`` and nothing it
produces may be reported as real model performance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from tests.synthetic import synthetic_dataset  # noqa: E402

SYNTHETIC_REGION = {"name": "SYNTHETIC test region", "max_distance_km": 25,
                    "points": [{"unit_id": f"u{i}", "latitude": 40.7 + 0.01 * i, "longitude": -74.0 + 0.01 * i} for i in range(3)]}
INSIDE = (40.71, -74.0)      # inside the synthetic region
OUTSIDE = (19.076, 72.8777)  # Mumbai: far outside it


@pytest.fixture(scope="session")
def trained_models_dir(tmp_path_factory):
    from app.ml.training import TrainConfig, run_training
    root = tmp_path_factory.mktemp("synthetic_model")
    df = synthetic_dataset(n_units=3, start="2021-01-01", end="2022-12-31", seed=1)
    csv = root / "synthetic.csv"
    df.to_csv(csv, index=False)
    manifest = root / "synthetic.manifest.json"
    manifest.write_text(json.dumps({"region": SYNTHETIC_REGION, "sources": [{"name": "SYNTHETIC test data", "licence": "n/a"}]}))
    models = root / "models"
    run_training(TrainConfig(dataset_path=csv, models_dir=models, manifest_path=manifest, test_start="2022-07-01",
                             cv_folds=2, min_class_count=20, log=lambda m: None))
    return models
