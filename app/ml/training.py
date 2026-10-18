"""Offline training / evaluation pipeline.

Run through ``scripts/train_model.py``. Nothing here is imported by the web
app at request time, so web start-up never trains or needs pandas.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app.core.features import FEATURE_SCHEMA_VERSION, FEATURE_SPECS, MODEL_FEATURES, RISK_CLASSES
from app.ml.dataset import WINDOW_HOURS

CLASS_ORDER = list(RISK_CLASSES)
REQUIRED_COLUMNS = ["timestamp_utc", "unit_id", "latitude", "longitude", *MODEL_FEATURES, "window_event_count"]


class TrainingAborted(Exception):
    """Raised when the data cannot support a valid, honest model. No artefacts are written."""


# --------------------------------------------------------------------------- data
def load_dataset(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise TrainingAborted(f"Dataset not found: {path}. Run scripts/prepare_dataset.py first.")
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TrainingAborted(f"Dataset is missing required columns: {missing}")
    report: dict[str, Any] = {"rows_in": int(len(df))}
    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in ["latitude", "longitude", *MODEL_FEATURES, "window_event_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unit_id"] = df["unit_id"].astype(str)

    n = len(df)
    df = df.dropna()
    report["dropped_missing_or_unparseable"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates(subset=["unit_id", "timestamp_utc"], keep="first")
    report["dropped_duplicate_unit_hour"] = n - len(df)

    n = len(df)
    ok = pd.Series(True, index=df.index)
    for name in MODEL_FEATURES:
        spec = FEATURE_SPECS[name]
        ok &= df[name].between(spec.minimum, spec.maximum)
    ok &= df["window_event_count"] >= 0
    ok &= df["window_event_count"] == df["window_event_count"].round()
    df = df[ok]
    report["dropped_out_of_range"] = n - len(df)

    n = len(df)
    bad = pd.Series(False, index=df.index)
    for shorter, longer in zip(MODEL_FEATURES[:3], MODEL_FEATURES[1:4]):
        bad |= df[longer] + 1e-6 < df[shorter]
    df = df[~bad]
    report["dropped_inconsistent_rain_windows"] = n - len(df)

    df = df.sort_values(["timestamp_utc", "unit_id"]).reset_index(drop=True)
    report["rows_out"] = int(len(df))
    if report["rows_out"] == 0:
        raise TrainingAborted("No usable rows remain after cleaning.")
    return df, report


def chronological_split(df: pd.DataFrame, test_start: str, purge_hours: int = WINDOW_HOURS):
    """Train = strictly before (test_start - purge). Test = on/after test_start.

    The purge removes rows whose forward label window would reach into the test period.
    """
    start = pd.Timestamp(test_start)
    train = df[df["timestamp_utc"] < start - pd.Timedelta(hours=purge_hours)]
    test = df[df["timestamp_utc"] >= start]
    if train.empty or test.empty:
        raise TrainingAborted(f"Chronological split at {test_start} leaves an empty train or test set.")
    return train.copy(), test.copy()


def thresholds_from_train(train_counts: pd.Series, high_quantile: float) -> dict[str, int]:
    positive = train_counts[train_counts >= 1]
    if positive.empty:
        raise TrainingAborted("The training period contains no incident reports; a supervised model cannot be trained.")
    high_min = max(2, int(np.ceil(np.quantile(positive, high_quantile))))
    return {"moderate_min": 1, "high_min": high_min, "high_quantile": float(high_quantile)}


def label_counts(counts, thresholds: dict[str, int]) -> np.ndarray:
    counts = np.asarray(counts)
    return np.select([counts >= thresholds["high_min"], counts >= thresholds["moderate_min"]],
                     ["High", "Moderate"], default="Low")


def class_distribution(labels) -> dict[str, int]:
    s = pd.Series(labels).value_counts()
    return {c: int(s.get(c, 0)) for c in CLASS_ORDER}


# ------------------------------------------------------------------------- models
def _preprocessor(scale: bool) -> ColumnTransformer:
    rain_idx = [i for i, f in enumerate(MODEL_FEATURES) if f.startswith("precip_")]
    other_idx = [i for i, f in enumerate(MODEL_FEATURES) if not f.startswith("precip_")]
    log = FunctionTransformer(np.log1p, validate=False)
    rain = Pipeline([("log1p", log), ("scale", StandardScaler())]) if scale else log
    other = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("rain", rain, rain_idx), ("other", other, other_idx)])


def candidate_factories(seed: int) -> dict[str, Callable[[], Pipeline]]:
    return {
        "logistic_regression": lambda: Pipeline([
            ("prep", _preprocessor(True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "random_forest": lambda: Pipeline([
            ("prep", _preprocessor(False)),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=50, max_samples=0.3,
                                           class_weight="balanced_subsample", n_jobs=-1, random_state=seed))]),
        "hist_gradient_boosting": lambda: Pipeline([
            ("prep", _preprocessor(False)),
            ("clf", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
                                                   min_samples_leaf=100, class_weight="balanced",
                                                   early_stopping=False, random_state=seed))]),
    }


def _X(df: pd.DataFrame) -> np.ndarray:
    return df[list(MODEL_FEATURES)].to_numpy(dtype=float)


def _ordered_proba(model, X) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    out = np.zeros((len(X), len(CLASS_ORDER)))
    for j, name in enumerate(CLASS_ORDER):
        if name in classes:
            out[:, j] = raw[:, classes.index(name)]
    return out


# ----------------------------------------------------------------------- metrics
def evaluate_predictions(y_true, y_pred, proba=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [c for c in CLASS_ORDER if c in set(y_true)]
    report = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "per_class": {c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]),
                          "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])}
                      for c in CLASS_ORDER},
        "labels": CLASS_ORDER,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
        "classes_present_in_truth": present,
    }
    if proba is not None:
        onehot = np.stack([(y_true == c).astype(float) for c in CLASS_ORDER], axis=1)
        out["brier_score"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        try:
            order = [CLASS_ORDER.index(c) for c in sorted(CLASS_ORDER)]   # sklearn expects sorted labels
            out["log_loss"] = float(log_loss(y_true, proba[:, order], labels=sorted(CLASS_ORDER)))
        except ValueError:
            out["log_loss"] = None
        out["high_reliability"] = reliability_table(onehot[:, 2], proba[:, 2])
    return out


def reliability_table(y_bin: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else None,
                     "observed_rate": float(y_bin[m].mean()) if m.any() else None})
    return rows


def fit_rain_rule(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Best single-feature rainfall-threshold rule (last-3-hours rain), chosen on TRAIN only."""
    best = None
    for t_mod in (0.1, 0.2, 0.5, 1, 2, 3, 5, 8):
        for t_high in (2, 5, 8, 10, 15, 20, 30, 50):
            if t_high <= t_mod:
                continue
            pred = np.where(x_train >= t_high, "High", np.where(x_train >= t_mod, "Moderate", "Low"))
            score = f1_score(y_train, pred, labels=CLASS_ORDER, average="macro", zero_division=0)
            if best is None or score > best["train_macro_f1"]:
                best = {"feature": "precip_past_3h_mm", "moderate_mm": t_mod, "high_mm": t_high,
                        "train_macro_f1": float(score)}
    return best


def apply_rain_rule(rule: dict, x: np.ndarray) -> np.ndarray:
    return np.where(x >= rule["high_mm"], "High", np.where(x >= rule["moderate_mm"], "Moderate", "Low"))


def forward_chain_folds(train: pd.DataFrame, n_folds: int, purge_hours: int):
    edges = pd.date_range(train["timestamp_utc"].min(), train["timestamp_utc"].max(), periods=n_folds + 2)
    ts = train["timestamp_utc"].to_numpy()
    for k in range(1, n_folds + 1):
        v0, v1 = edges[k].to_datetime64(), edges[k + 1].to_datetime64()
        tr = np.where(ts < v0 - np.timedelta64(purge_hours, "h"))[0]
        va = np.where((ts >= v0) & ((ts <= v1) if k == n_folds else (ts < v1)))[0]
        yield tr, va


# -------------------------------------------------------------------------- run
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    raise TypeError(f"not serialisable: {type(o)}")


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class TrainConfig:
    dataset_path: Path
    models_dir: Path
    manifest_path: Path | None = None
    test_start: str = "2024-01-01"
    cv_folds: int = 3
    high_quantile: float = 0.90
    min_class_count: int = 100
    seed: int = 42
    log: Callable[[str], None] = print


def run_training(cfg: TrainConfig) -> dict:
    log = cfg.log
    manifest = {}
    if cfg.manifest_path and Path(cfg.manifest_path).exists():
        manifest = json.loads(Path(cfg.manifest_path).read_text(encoding="utf-8"))
    else:
        log("WARNING: no dataset manifest found; source/licence fields will say 'unknown'.")

    raw = load_dataset(cfg.dataset_path)
    df, data_report = clean_dataset(raw)
    log(f"Cleaned dataset: {data_report}")

    train, test = chronological_split(df, cfg.test_start, WINDOW_HOURS)
    thresholds = thresholds_from_train(train["window_event_count"], cfg.high_quantile)
    y_train = label_counts(train["window_event_count"], thresholds)
    y_test = label_counts(test["window_event_count"], thresholds)
    dist = {"train": class_distribution(y_train), "test": class_distribution(y_test)}
    log(f"Thresholds (from training period only): {thresholds}")
    log(f"Class distribution: {dist}")

    for split, d in dist.items():
        short = [c for c, n in d.items() if n < cfg.min_class_count]
        if short:
            raise TrainingAborted(
                f"Class(es) {short} have fewer than {cfg.min_class_count} examples in the {split} split ({d}). "
                "A reliable classifier cannot be trained or evaluated; add more labelled observations "
                "(see data/README.md). No model was saved.")

    factories = candidate_factories(cfg.seed)
    Xtr, Xte = _X(train), _X(test)

    # ---- model selection: forward-chaining CV inside the training period only
    cv: dict[str, dict] = {}
    for name, make in factories.items():
        scores, rec_high = [], []
        for tr_idx, va_idx in forward_chain_folds(train, cfg.cv_folds, WINDOW_HOURS):
            if len(set(y_train[tr_idx])) < 2 or len(va_idx) == 0:
                continue
            model = make().fit(Xtr[tr_idx], y_train[tr_idx])
            pred = model.predict(Xtr[va_idx])
            m = evaluate_predictions(y_train[va_idx], pred)
            scores.append(m["macro_f1"])
            rec_high.append(m["per_class"]["High"]["recall"])
        if not scores:
            raise TrainingAborted("Cross-validation could not be run (training period too short).")
        cv[name] = {"fold_macro_f1": scores, "mean_macro_f1": float(np.mean(scores)),
                    "std_macro_f1": float(np.std(scores)), "mean_recall_high": float(np.mean(rec_high))}
        log(f"CV {name}: macro-F1 {cv[name]['mean_macro_f1']:.3f} +/- {cv[name]['std_macro_f1']:.3f}, "
            f"High recall {cv[name]['mean_recall_high']:.3f}")
    selected = max(cv, key=lambda n: (cv[n]["mean_macro_f1"], cv[n]["mean_recall_high"]))
    log(f"Selected: {selected}")

    # ---- final fit on the training period; evaluate once on the untouched test period
    final = factories[selected]().fit(Xtr, y_train)
    pred_te = final.predict(Xte)
    proba_te = _ordered_proba(final, Xte)
    test_metrics = evaluate_predictions(y_test, pred_te, proba_te)
    wet = (test["precip_past_24h_mm"].to_numpy() >= 1.0)
    wet_metrics = evaluate_predictions(y_test[wet], pred_te[wet]) if wet.sum() >= 30 else None

    majority = pd.Series(y_train).mode()[0]
    baselines = {
        "majority_class": {"predicts": str(majority),
                           **evaluate_predictions(y_test, np.full(len(y_test), majority))},
    }
    rule = fit_rain_rule(train["precip_past_3h_mm"].to_numpy(), y_train)
    baselines["rainfall_rule"] = {**rule, **evaluate_predictions(
        y_test, apply_rain_rule(rule, test["precip_past_3h_mm"].to_numpy()))}

    # ---- location-aware check: hold one unit out entirely (train period vs. its test period)
    location = None
    units = sorted(df["unit_id"].unique())
    if len(units) >= 3:
        per_unit = {}
        for u in units:
            tr_m = (train["unit_id"] != u).to_numpy()
            te_m = (test["unit_id"] == u).to_numpy()
            if te_m.sum() < 100 or len(set(y_train[tr_m])) < 3:
                continue
            m = factories[selected]().fit(Xtr[tr_m], y_train[tr_m])
            r = evaluate_predictions(y_test[te_m], m.predict(Xte[te_m]))
            per_unit[u] = {"macro_f1": r["macro_f1"], "recall_high": r["per_class"]["High"]["recall"], "n": r["n"]}
        if per_unit:
            location = {"per_unit": per_unit,
                        "mean_macro_f1": float(np.mean([v["macro_f1"] for v in per_unit.values()])),
                        "note": "Each unit is held out of training entirely and scored on the test period. "
                                "Units that share a weather grid cell are not independent."}

    # ---- permutation importance (global, on the test period)
    rng = np.random.default_rng(cfg.seed)
    take = rng.choice(len(Xte), size=min(20000, len(Xte)), replace=False)
    pi = permutation_importance(final, Xte[take], y_test[take], scoring="f1_macro", n_repeats=5,
                                random_state=cfg.seed, n_jobs=1)
    importance = {f: {"mean_drop_in_macro_f1": float(pi.importances_mean[i]), "std": float(pi.importances_std[i])}
                  for i, f in enumerate(MODEL_FEATURES)}

    # ---- persist (model first, then metadata that contains its hash)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    model_path = cfg.models_dir / "waterlogging_model.joblib"
    tmp_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(final, tmp_model, compress=3)
    os.replace(tmp_model, model_path)
    sha = _sha256(model_path)
    version = f"{now:%Y%m%d-%H%M%S}-{sha[:8]}"

    region = manifest.get("region") or {
        "name": "Unknown (no manifest)",
        "points": [{"unit_id": u, "latitude": float(g["latitude"].iloc[0]), "longitude": float(g["longitude"].iloc[0])}
                   for u, g in df.groupby("unit_id")],
        "max_distance_km": 25,
    }
    counts = {"rows_raw": data_report["rows_in"], "rows_clean": data_report["rows_out"],
              "train_records": int(len(train)), "test_records": int(len(test)),
              "units": units}
    ts_range = {"first_utc": df["timestamp_utc"].min(), "last_utc": df["timestamp_utc"].max(),
                "train_last_utc": train["timestamp_utc"].max(), "test_first_utc": test["timestamp_utc"].min()}
    metadata = {
        "model_version": version, "model_name": selected, "trained_at_utc": now.isoformat(),
        "model_file": model_path.name, "model_sha256": sha,
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "features": list(MODEL_FEATURES),
        "feature_specs": {f: {"label": FEATURE_SPECS[f].label, "unit": FEATURE_SPECS[f].unit} for f in MODEL_FEATURES},
        "classes": list(final.classes_), "class_order": CLASS_ORDER,
        "target": {
            "name": "reported_waterlogging_level_next_6h",
            "definition": (f"Number of observed incident reports created in the {WINDOW_HOURS} hours starting at the "
                           "feature timestamp, for the same unit. Low = 0 reports; Moderate = "
                           f"{thresholds['moderate_min']}..{thresholds['high_min'] - 1}; High = >= {thresholds['high_min']}."),
            "thresholds": thresholds,
            "threshold_rule": f"High threshold = {int(cfg.high_quantile * 100)}th percentile of non-zero window counts, "
                              "computed on the training period only.",
        },
        "dataset": {"manifest": manifest or None, "file": Path(cfg.dataset_path).name,
                    "file_sha256": _sha256(Path(cfg.dataset_path)), "counts": counts, "time_range": ts_range,
                    "cleaning_report": data_report},
        "split": {"strategy": "chronological hold-out", "test_start": cfg.test_start,
                  "purge_hours": WINDOW_HOURS,
                  "description": f"Train on rows before {cfg.test_start} minus a {WINDOW_HOURS}-hour purge; test on "
                                 "rows on/after it. Model selection used forward-chaining CV inside the training "
                                 "period. Preprocessing is part of the fitted pipeline (fit on training rows only)."},
        "selection": {"metric": "mean macro-F1 over forward-chaining CV folds (tie-break: recall of High)",
                      "cv_folds": cfg.cv_folds, "candidates": cv, "selected": selected},
        "applicability": region,
        "feature_importance": importance,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
        "random_seed": cfg.seed,
        "scores_are_calibrated_probabilities": False,
        "limitations": [
            "Labels are reported incidents (e.g. citizen complaints), which reflect reporting behaviour as well as water on the street.",
            "Rainfall comes from a gridded reanalysis at training time and from a forecast model at prediction time; the two differ.",
            "The model only knows weather. Drainage capacity, terrain, tides, land cover and maintenance are not inputs.",
            "The model is only valid in the region listed under 'applicability'.",
            "Offline test metrics do not guarantee real-world accuracy.",
        ],
    }
    metrics = {
        "evaluated_at_utc": now.isoformat(), "model_version": version, "model_name": selected,
        "test": test_metrics, "test_wet_hours": wet_metrics, "baselines": baselines,
        "cross_validation": {"strategy": f"forward-chaining, {cfg.cv_folds} folds, {WINDOW_HOURS} h purge",
                             "candidates": cv, "selected": selected},
        "location_holdout": location,
        "class_distribution": dist, "counts": counts, "time_range": ts_range,
        "per_year_positive_rate": _per_year_rate(df, thresholds),
    }
    _atomic_write_json(cfg.models_dir / "model_metadata.json", metadata)
    _atomic_write_json(cfg.models_dir / "model_metrics.json", metrics)
    archive = cfg.models_dir / "archive" / version
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("waterlogging_model.joblib", "model_metadata.json", "model_metrics.json"):
        shutil.copy2(cfg.models_dir / name, archive / name)
    log(f"Saved model {version} to {cfg.models_dir}")
    return {"metadata": metadata, "metrics": metrics}


def _per_year_rate(df: pd.DataFrame, thresholds: dict) -> dict:
    labels = label_counts(df["window_event_count"], thresholds)
    years = df["timestamp_utc"].dt.year.to_numpy()
    out = {}
    for y in sorted(set(years)):
        m = years == y
        out[str(int(y))] = {"rows": int(m.sum()), **{c: float(np.mean(labels[m] == c)) for c in CLASS_ORDER}}
    return out
