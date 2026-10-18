"""HTML pages. Every page extends ``templates/base.html`` (shared navigation + footer)."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from app.content import DEVELOPER, PLACEHOLDERS
from app.core.features import FEATURE_SPECS, HEURISTIC_FEATURES, MODEL_FEATURES, RISK_CLASSES
from app.schemas import parse_manual_form
from app.services.prediction_service import METADATA_FILE, METRICS_FILE, NoPredictionPossible, read_json_file
from app.services.runner import predict_and_record

router = APIRouter()

MANUAL_GROUPS = (
    ("Rain that has already fallen", ("precip_current_mm", "precip_past_3h_mm", "precip_past_24h_mm", "precip_past_72h_mm")),
    ("Forecast rain", ("precip_next_6h_mm", "rain_probability_pct")),
    ("Other conditions", ("temperature_c", "humidity_pct", "wind_speed_kmh")),
)


def render(request: Request, name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(request=request, name=name, context=context, status_code=status_code)


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render(request, "index.html", active="home")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    s = request.app.state
    stats = s.db.stats()
    show_distribution = stats["total"] >= 5
    return render(request, "dashboard.html", active="dashboard", model=s.predictor.state.public(),
                  coverage=s.predictor.coverage, history_available=s.db.available, history=s.db.recent(10),
                  stats=stats, show_distribution=show_distribution, classes=RISK_CLASSES,
                  last_weather_success=s.weather.last_success_at, last_weather_error=s.weather.last_error)


def _manual_context(request: Request, **extra: Any) -> dict[str, Any]:
    predictor = request.app.state.predictor
    return {"active": "predict", "groups": MANUAL_GROUPS, "specs": FEATURE_SPECS, "model_fields": set(MODEL_FEATURES),
            "model": predictor.state.public(), "coverage": predictor.coverage,
            "errors": {}, "values": {"model_area": "other"}, "result": None, **extra}


@router.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):
    return render(request, "predict.html", **_manual_context(request))


@router.post("/predict", response_class=HTMLResponse)
async def predict_submit(request: Request):
    limiter = request.app.state.limiter
    retry = limiter.check(request.client.host if request.client else "unknown")
    if retry is not None:
        return render(request, "predict.html", 429, **_manual_context(
            request, errors={"__all__": f"Too many requests. Please wait about {int(retry)} seconds and try again."}))
    form = await request.form()
    parsed, errors, raw = parse_manual_form(form)
    if parsed is None:
        return render(request, "predict.html", 422, **_manual_context(request, errors=errors, values=raw))
    try:
        result = await predict_and_record(request.app.state, inputs=parsed.to_inputs(), source="manual",
                                          area_confirmed=parsed.model_area == "covered")
    except NoPredictionPossible as exc:
        return render(request, "predict.html", 422, **_manual_context(request, errors={"__all__": str(exc)}, values=raw))
    return render(request, "predict.html", **_manual_context(request, values=raw, result=result))


@router.get("/how-it-works", response_class=HTMLResponse)
def how_it_works(request: Request):
    predictor = request.app.state.predictor
    meta = predictor.metadata or {}
    return render(request, "how_it_works.html", active="how", model=predictor.state.public(), coverage=predictor.coverage,
                  model_features=[FEATURE_SPECS[n] for n in MODEL_FEATURES],
                  fallback_features=[FEATURE_SPECS[n] for n in HEURISTIC_FEATURES],
                  target=(meta.get("target") or {}).get("definition"))


def _cm_rows(test: dict) -> list[dict]:
    labels = test.get("labels") or list(RISK_CLASSES)
    matrix = test.get("confusion_matrix") or []
    rows = []
    for i, row in enumerate(matrix):
        total = sum(row) or 0
        cells = []
        for n in row:
            share = (n / total) if total else 0.0
            cells.append({"n": int(n), "share": round(100 * share, 1), "level": min(5, int(share * 5.999))})
        rows.append({"label": labels[i] if i < len(labels) else str(i), "total": int(total), "cells": cells})
    return rows


@router.get("/model-performance", response_class=HTMLResponse)
def model_performance(request: Request):
    s = request.app.state
    models_dir = s.predictor.models_dir
    metrics = read_json_file(models_dir / METRICS_FILE)
    meta = read_json_file(models_dir / METADATA_FILE)
    state = s.predictor.state.public()
    consistent = bool(metrics and meta and metrics.get("model_version") == meta.get("model_version"))
    test = (metrics or {}).get("test") or {}
    return render(request, "model_performance.html", active="performance", state=state,
                  metrics=metrics if consistent else None, meta=meta if consistent else None,
                  inconsistent=bool(metrics and meta and not consistent),
                  cm_rows=_cm_rows(test) if consistent else [], classes=RISK_CLASSES)


@router.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return render(request, "about.html", active="about", dev=DEVELOPER, placeholders=PLACEHOLDERS)
