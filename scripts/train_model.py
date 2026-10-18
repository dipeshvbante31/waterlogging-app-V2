#!/usr/bin/env python
"""Train, validate and save the waterlogging model. Run offline, never at web start-up.

    python scripts/train_model.py
    python scripts/train_model.py --dataset data/processed/nyc_street_flooding_reports.csv --test-start 2025-01-01

Nothing is written to models/ unless the data can support an honest model (enough observed events in every
class, in both the training and the test period). Metrics are computed on a later, untouched test period.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml.training import TrainConfig, TrainingAborted, run_training  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "processed" / "nyc_street_flooding_reports.csv")
    p.add_argument("--manifest", type=Path, default=None, help="default: <dataset>.manifest.json next to the CSV")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--test-start", default="2025-01-01", help="first UTC date of the held-out test period")
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--high-quantile", type=float, default=0.90)
    p.add_argument("--min-class-count", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    manifest = a.manifest or a.dataset.with_suffix(".manifest.json")
    cfg = TrainConfig(dataset_path=a.dataset, models_dir=a.models_dir, manifest_path=manifest, test_start=a.test_start,
                      cv_folds=a.cv_folds, high_quantile=a.high_quantile, min_class_count=a.min_class_count, seed=a.seed)
    try:
        out = run_training(cfg)
    except TrainingAborted as exc:
        print(f"\nTRAINING STOPPED - no model was saved.\n{exc}", file=sys.stderr)
        return 2
    t = out["metrics"]["test"]
    print("\n=== Test-period results (computed from this run) ===")
    print(f"model: {out['metadata']['model_name']}   version: {out['metadata']['model_version']}")
    print(f"macro F1: {t['macro_f1']:.3f}   accuracy: {t['accuracy']:.3f}   High recall: {t['per_class']['High']['recall']:.3f}")
    for name, b in out["metrics"]["baselines"].items():
        print(f"baseline {name}: macro F1 {b['macro_f1']:.3f}")
    print("Restart the web app to load the new model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
