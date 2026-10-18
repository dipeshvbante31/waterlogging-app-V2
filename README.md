# Waterlogging Watch
<div align="center">
🌧️ Waterlogging Watch
Weather-driven waterlogging risk screening powered by FastAPI and machine learning
A final-year BE Computer Science & Engineering project by Dipesh Vishnu Bante
</div>

A FastAPI web application that estimates short-term **waterlogging risk** from live weather, with an offline pipeline that
trains a machine-learning model from real historical data. Final-year BE (Computer Science and Engineering) project by
Dipesh Vishnu Bante.

> **It is an informational screening tool, not an official flood warning.** Follow local authorities and emergency instructions.

## What you need to know first (honest status)

* The **application, training pipeline and tests are complete**, but **no trained model is shipped**. A model only exists after *you*
  run the two scripts below on a machine with internet access. Until then every prediction is explicitly labelled
  **"Weather-based fallback (heuristic)"** and the Model Performance page says metrics are unavailable. Nothing is faked.
* **No open, dated record of observed waterlogging for Mumbai (or any other Indian city) was found.** Rainfall records alone are not
  waterlogging labels. The default training data is therefore **New York City**: Open-Meteo hourly historical weather +
  NYC 311 street-flooding complaints (observed reports). A model trained on it is used **only inside its coverage area (NYC)**;
  anywhere else, including Mumbai, the labelled fallback is used. To train for another city you need dated incident records for it
  (see [Training for another area](#training-for-another-area)).
* Offline validation metrics do not guarantee real-world accuracy.

## Features

| Page | Route | Purpose |
|---|---|---|
| Live Weather (home) | `/` | Search a place by name, see current weather, past-rain totals, next-6-hour forecast, risk category, method used and data freshness (live / cached / stale) |
| Dashboard | `/dashboard` | Last checked place, model status, data freshness, saved prediction history and distribution (only from real saved records; empty states otherwise) |
| Manual Prediction | `/predict` | Enter weather values yourself; validated on the browser and the server; model or labelled fallback |
| How It Works | `/how-it-works` | Workflow, factors used / not used, forecast vs risk estimate vs confirmed waterlogging |
| Model Performance | `/model-performance` | Metrics read from the real training run; honest empty state if none |
| About | `/about` | Developer details and editable placeholders (`app/content.py`) |

Also: JSON API (`POST /api/predict`, `POST /api/predict/manual`, `GET /api/model`), `GET /health`, `GET /ready`, light/dark theme,
responsive layout. **No maps and no CSV upload** exist anywhere in the site; CSV files are only read locally by the training scripts.

## How predictions are made

1. A trained model is used only if **all** are true: the artefacts pass checks (fixed file names in `WW_MODELS_DIR`, SHA-256
   matches `model_metadata.json`, feature schema and scikit-learn version match), every model input is available, and the location is
   inside the model's coverage area (for manual input, the user states the readings are for that area).
2. Otherwise a transparent **heuristic** (0–100 *weather risk index*, not a probability) is used, and the result says why
   (`model_missing`, `outside_region`, `missing_features`, ...).
3. If the model raises an error at prediction time it is logged and the result is labelled a fallback with reason `model_error`;
   it is never presented as a model prediction.

Model class scores are **uncalibrated** and are never shown as probabilities of waterlogging.

**Model inputs (7):** rain in the last 1 h, 3 h, 24 h, 72 h; temperature; relative humidity; wind speed.
**Fallback inputs:** current rain, next-6-hour forecast rain, peak rain probability.
**Not used:** drainage, terrain/elevation, tides, soil/land cover, local infrastructure.

## Folder structure

```
main.py                     entry point (python -m uvicorn main:app)
app/
  config.py                 settings from environment variables
  web.py                    app factory, middleware, error handlers
  routes/pages.py, api.py   HTML pages and JSON API
  schemas.py                Pydantic request models + form parsing
  services/                 weather_service (Open-Meteo), prediction_service (model/fallback), runner
  core/                     feature schema, forecast parser, heuristic, geo helpers (shared by training and serving)
  ml/                       dataset.py, sources.py, training.py  (offline only; needs pandas)
  db.py                     SQLAlchemy prediction history
  ratelimit.py, content.py
templates/  static/         Jinja2 pages, CSS, vanilla JS, logo
scripts/prepare_dataset.py  download + assemble the training table
scripts/train_model.py      compare models, evaluate, save artefacts
models/  data/              artefacts and working data (git-ignored; see their READMEs)
tests/                      pytest suite
```

## Installation

### Windows (PowerShell or cmd) — no virtual environment required
```
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000. (A virtual environment is optional: `python -m venv .venv` then `.venv\Scripts\activate`.)

### Linux / macOS
```
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Python 3.10+ is required (developed on 3.12). Interactive API docs are at `/docs` in development.

## Getting data and training the model

Training needs extra packages (`pandas`) and internet access, and is **never** done at web start-up.
```
python -m pip install -r requirements-train.txt

# 1. (optional) check how 311 labels flooding, then download weather + observed reports (~10 years, five NYC points)
python scripts/prepare_dataset.py --list-descriptors
python scripts/prepare_dataset.py                      # writes data/processed/nyc_street_flooding_reports.csv (+ .manifest.json)

# 2. train, validate on a later untouched period, and save the model
python scripts/train_model.py                          # writes models/waterlogging_model.joblib, model_metadata.json, model_metrics.json

# 3. restart the web app; /model-performance now shows the real metrics
```
Useful options: `--start/--end` (prepare), `--test-start`, `--cv-folds`, `--high-quantile`, `--min-class-count` (train). Defaults
use data from 2016-01-01 to 2026-09-30 with 2025-01-01 onward held out as the test period. The 311 filter defaults
(`complaint_type=Sewer`, descriptor like `Street Flooding%`) should be confirmed with `--list-descriptors` before a large download.

### Data sources and licences (check current terms yourself)

| Source | Used for | Terms |
|---|---|---|
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (hourly ERA5-based reanalysis) | training weather features | Data CC BY 4.0; free API is for **non-commercial** use. Attribution is shown in the footer. |
| [Open-Meteo Forecast + Geocoding APIs](https://open-meteo.com/en/docs) | live weather and place search | Same: free tier non-commercial, rate-limited; commercial or heavy use needs a paid plan or self-hosting |
| [NYC Open Data – 311 Service Requests](https://data.cityofnewyork.us/) | observed outcome labels (street-flooding complaints) | NYC Open Data terms of use. Not redistributed by this repository. |

The scripts download data to your machine only (`data/` is git-ignored). Licences above were recorded from the providers' public
documentation; re-check them before publishing or commercial use.

### Target definition (what a "positive" means)

For each hour *t* and each location unit: the number of **observed street-flooding reports created in [t, t+6 h)**.
Classes are derived from the **training period only**: `Low` = 0 reports, `Moderate` = at least 1, `High` = at least the 90th percentile of the
non-zero counts. Features use only rain that fell **up to and including** hour *t* (no future information); training rows within 6 h of the
test period are purged. Negatives (`Low`) are hours with zero reports, which also includes places where no one reported.

### Training CSV schema

| column | meaning |
|---|---|
| `timestamp_utc` | hour start (UTC) |
| `unit_id`, `latitude`, `longitude` | location unit (e.g. borough centre) |
| `precip_current_mm`, `precip_past_3h_mm`, `precip_past_24h_mm`, `precip_past_72h_mm` | trailing rain totals (mm) |
| `temperature_c`, `humidity_pct`, `wind_speed_kmh` | conditions at the hour |
| `window_event_count` | observed reports in the next 6 h (the target is derived from this) |

### Evaluation methodology

* Chronological hold-out: train strictly before the test start, test afterwards (6 h purge).
* Model selection by **macro F1** on forward-chaining validation folds *inside* the training period; the test period is scored once.
* Candidates: logistic regression (baseline), random forest, histogram gradient boosting. Preprocessing is fitted on training rows only.
* Reported next to the model: accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix, two baselines
  (most-common class; single rainfall threshold), permutation importance, a leave-one-location-out check, Brier score / log loss.
* Training **aborts without saving** if the data lack enough events per class in either period.

### Updating the model

When new *verified* observations exist, re-run `prepare_dataset.py` with a later `--end`, then `train_model.py`. Every trained version is also
copied to `models/archive/<version>/`. The web app never retrains itself and never learns from user-entered values.

### Training for another area

You need dated incident records (for example municipal waterlogging complaints for Mumbai obtained with permission). Provide
`events.csv` (`timestamp`, `unit_id`) and `units.csv` (`unit_id`, `latitude`, `longitude`), then:
```
python scripts/prepare_dataset.py --source custom --events-csv events.csv --units-csv units.csv --events-tz Asia/Kolkata \
    --region-name "Mumbai" --events-source "who supplied it, when" --events-licence "the terms of use" --start 2019-01-01 --end 2025-12-31
python scripts/train_model.py --dataset data/processed/custom_incident_reports.csv --test-start 2025-01-01
```
Do not use news articles, rainfall-only records or invented incidents as labels.

## Running

* **With a trained model:** train as above, start the app; `/dashboard` shows "Trained model in use". It is applied only inside its coverage area.
* **Without one:** just start the app. Everything works; results are labelled "Weather-based fallback (heuristic)".

## Configuration (environment variables; see `.env.example`)

| variable | default | meaning |
|---|---|---|
| `WW_ENV` | `development` | `production` hides `/docs` |
| `WW_DATABASE_URL` | `sqlite:///./data/waterlogging.db` | history database (`postgresql://...` for production) |
| `WW_SAVE_HISTORY` | `true` | store predictions (weather numbers, coarse place name, rounded coordinates; no IPs) |
| `WW_MODELS_DIR` | `models` | where artefacts are loaded from |
| `WW_HTTP_TIMEOUT_S` | `10` | provider timeout |
| `WW_WEATHER_CACHE_TTL_S` | `600` | cache lifetime ("cached" label) |
| `WW_WEATHER_STALE_TTL_S` | `10800` | max age of data shown (labelled "stale") if the provider fails |
| `WW_RATE_LIMIT_PER_MINUTE` | `30` | per-client limit on prediction endpoints (per process; `0` = off) |
| `WW_MAX_BODY_BYTES` | `16384` | request size limit |
| `WW_LOG_LEVEL` | `INFO` | logging level |

Load a file with `python -m uvicorn main:app --env-file .env`.

## Testing

```
python -m pip install -r requirements-dev.txt
python -m pytest
```
Tests mock all external APIs (no internet needed). The model-loading and labelling tests train a model on **synthetic data in a temporary
directory**; it exists only to exercise code paths and is never written to `models/` or shown as real performance. Covered: all pages and
navigation, location and manual-input validation, model loading/missing/corrupt/incompatible artefacts, model-vs-fallback labelling,
weather timeouts / rate limits / missing fields / stale data, history and database initialisation, rate and size limits, training
safeguards (no leakage, abort on insufficient data).

## Deployment

1. Train the model on a trusted machine and place `models/` next to the app (or mount it and set `WW_MODELS_DIR`).
2. Use **PostgreSQL** (`pip install -r requirements-prod.txt`, `WW_DATABASE_URL=postgresql://...`). Tables are created on start-up with
   `CREATE TABLE IF NOT EXISTS`; there are no migration scripts yet (add Alembic if you change the schema). Do not rely on SQLite on hosts
   with an ephemeral disk.
3. Run with several workers behind a TLS-terminating reverse proxy:
   `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2 --proxy-headers`
   or build the supplied `Dockerfile` (`docker build -t waterlogging-watch .`, `docker run -p 8000:8000 -e WW_DATABASE_URL=... -v ./models:/srv/app/models waterlogging-watch`).
4. Set `WW_ENV=production`. Probes: `GET /health` (liveness), `GET /ready` (model/database/provider status; degraded features do not make it unready).
5. The in-process rate limiter is per worker; apply a real limit at the proxy for public deployments.
6. **Licensing:** Open-Meteo's free tier is for non-commercial use. A commercial deployment needs their paid API or self-hosted data.

Security measures present: input validation (Pydantic + range checks), fixed model path with checksum verification, CSP and other security
headers, body-size and rate limits, generic 500 responses, no secrets in the repo. This is not a security audit; do not describe it as enterprise-grade.

## Limitations

* Weather grids and reanalysis are not street-level sensors; drainage, blockages, terrain, tides and works are unknown.
* The training label is a count of citizen reports: it reflects reporting behaviour as well as actual water on streets, and report volumes change over time.
* Boroughs/units share weather grid cells, so rows are not independent; the held-out period and the leave-one-location-out check partly address this.
* Training uses reanalysis weather, serving uses forecast-model weather; small differences are possible.
* A model trained on one city is not valid for another; the app enforces a coverage area.
* The fallback thresholds are hand-written and unvalidated against observations.
* Live performance is not monitored automatically. Saved history records what was predicted, not what happened.

## Responsible use

Treat every result as an estimate. Use official weather and flood warnings, and follow emergency services, when making safety decisions.


© Copyright and Licensing
Copyright © 2026 Dipesh Vishnu Bante. All rights reserved.
The original Waterlogging Watch project code, application structure, interface implementation, documentation, and project-specific materials are the work of Dipesh Vishnu Bante, except where otherwise stated.
- No permission is granted to copy, republish, redistribute, or commercially use the original project code or materials without prior written permission from the copyright holder.
- This notice does not claim ownership of third-party software, libraries, APIs, datasets, trademarks, or other materials referenced by this project. Those remain subject to their respective owners' licences and terms.
- Open-Meteo data/API usage and NYC Open Data usage are subject to the terms described in Data Sources and Licences. Review current provider terms before redistribution or commercial use.
- If you want to reuse this project, contact the copyright holder and obtain permission first.
Project author: Dipesh Vishnu Bante
GitHub: github.com/dipeshvbante31