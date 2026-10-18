"""JSON API. ``POST /api/predict`` keeps the original contract (``{"location": "..."}``) and adds freshness info."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import APP_VERSION
from app.content import DISCLAIMER
from app.core.features import FEATURE_SPECS, MODEL_FEATURES
from app.schemas import LocationRequest, ManualInput
from app.services.prediction_service import NoPredictionPossible
from app.services.runner import predict_and_record

router = APIRouter()


async def rate_limit(request: Request) -> None:
    retry = request.app.state.limiter.check(request.client.host if request.client else "unknown")
    if retry is not None:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down and try again shortly.",
                            headers={"Retry-After": str(int(retry))})


def _r(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(value, digits)


@router.post("/api/predict")
async def predict_live(payload: LocationRequest, request: Request, _: None = Depends(rate_limit)) -> dict[str, Any]:
    state = request.app.state
    weather = await state.weather.get_weather(payload.location)         # WeatherError -> JSON error handler
    snap, place, i = weather.snapshot, weather.place, weather.snapshot.inputs
    stamp = f"{snap.observed_at} ({snap.timezone})"
    try:
        prediction = await predict_and_record(state, inputs=i, source="live", latitude=place.latitude, longitude=place.longitude,
                                              data_timestamp=stamp, location_name=place.name)
    except NoPredictionPossible:
        raise HTTPException(status_code=503, detail="The weather service did not return enough rainfall data to estimate risk. "
                                                    "Please try again shortly.")
    if weather.status != "live":
        prediction.setdefault("notes", []).append(
            "Weather data is not live (" + weather.status + "), so this estimate reflects older conditions.")
    return {
        "location": {"name": place.name, "admin1": place.admin1, "country": place.country,
                     "latitude": place.latitude, "longitude": place.longitude},
        "weather": {
            "temperature_c": _r(i.temperature_c), "humidity_pct": _r(i.humidity_pct, 0),
            "current_precipitation_mm": _r(i.precip_current_mm), "wind_speed_kmh": _r(i.wind_speed_kmh),
            "past_3h_precipitation_mm": _r(i.precip_past_3h_mm), "past_24h_precipitation_mm": _r(i.precip_past_24h_mm),
            "past_72h_precipitation_mm": _r(i.precip_past_72h_mm),
            "next_6h_precipitation_mm": _r(i.precip_next_6h_mm), "next_6h_rain_probability_pct": _r(i.rain_probability_pct, 0),
            "observed_at": snap.observed_at, "timezone": snap.timezone, "missing_fields": snap.missing_fields,
        },
        "hourly_next": snap.hourly_next,
        "data": {"status": weather.status, "fetched_at": weather.fetched_at.isoformat(), "age_seconds": weather.age_seconds,
                 "note": weather.note, "provider": "Open-Meteo"},
        "prediction": prediction,
        "generated_at": prediction["generated_at"],
        "disclaimer": DISCLAIMER,
    }


@router.post("/api/predict/manual")
async def predict_manual(payload: ManualInput, request: Request, _: None = Depends(rate_limit)) -> dict[str, Any]:
    try:
        prediction = await predict_and_record(request.app.state, inputs=payload.to_inputs(), source="manual",
                                              area_confirmed=payload.model_area == "covered")
    except NoPredictionPossible as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"prediction": prediction, "disclaimer": DISCLAIMER}


@router.get("/api/model")
def model_status(request: Request) -> dict[str, Any]:
    predictor = request.app.state.predictor
    meta = predictor.metadata or {}
    return {"model": predictor.state.public(), "features": list(MODEL_FEATURES),
            "feature_units": {n: FEATURE_SPECS[n].unit for n in MODEL_FEATURES},
            "coverage": predictor.coverage, "classes": meta.get("classes"),
            "target": (meta.get("target") or {}).get("definition")}


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Does not touch external services."""
    return {"status": "ok", "version": APP_VERSION}


@router.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness: the app can serve. A missing model or database degrades features but does not make the app unready."""
    s = request.app.state
    body = {"status": "ready", "version": APP_VERSION, "model": s.predictor.state.public(),
            "database": {"available": s.db.available, "error": s.db.error},
            "weather": {"last_success_at": s.weather.last_success_at.isoformat() if s.weather.last_success_at else None,
                        "last_error": s.weather.last_error}}
    return JSONResponse(body, status_code=200)
