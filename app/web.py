"""Application factory: wires settings, services, middleware, error handlers and routers."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import APP_VERSION, BASE_DIR, Settings, load_settings
from app.content import COPYRIGHT, DISCLAIMER
from app.db import Database
from app.ratelimit import RateLimiter
from app.routes import api, pages
from app.services.prediction_service import PredictionService
from app.services.weather_service import WeatherError, WeatherService

log = logging.getLogger("waterlogging")

NAV = (
    ("home", "/", "Live Weather"),
    ("dashboard", "/dashboard", "Dashboard"),
    ("predict", "/predict", "Manual Prediction"),
    ("how", "/how-it-works", "How It Works"),
    ("performance", "/model-performance", "Model Performance"),
    ("about", "/about", "About"),
)

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
       "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'")


def _fmt_dt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")


def _error_body(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, **extra}, "detail": message}


def create_app(settings: Optional[Settings] = None, *, weather: Optional[WeatherService] = None,
               predictor: Optional[PredictionService] = None, db: Optional[Database] = None) -> FastAPI:
    settings = settings or load_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await app.state.weather.aclose()
        app.state.db.dispose()

    app = FastAPI(title="Waterlogging Watch", version=APP_VERSION, lifespan=lifespan,
                  description="Waterlogging risk screening from live weather, with an optional trained ML model.",
                  docs_url=None if settings.is_production else "/docs",
                  redoc_url=None, openapi_url=None if settings.is_production else "/openapi.json")
    s = app.state
    s.settings = settings
    s.weather = weather or WeatherService(settings)
    s.predictor = predictor or PredictionService(settings.models_dir)
    s.db = db or Database(settings.database_url)
    s.db.init_schema()
    s.limiter = RateLimiter(settings.rate_limit_per_minute)

    templates = Jinja2Templates(directory=BASE_DIR / "templates")
    templates.env.filters["fmt_dt"] = _fmt_dt
    templates.env.globals.update(nav=NAV, copyright=COPYRIGHT, disclaimer=DISCLAIMER, app_version=APP_VERSION)
    s.templates = templates
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_body_bytes:
            return JSONResponse(_error_body("payload_too_large", "Request body is too large."), status_code=413)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if not request.url.path.startswith(("/docs", "/openapi")):
            response.headers.setdefault("Content-Security-Policy", CSP)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(WeatherError)
    async def weather_error(request: Request, exc: WeatherError):
        return JSONResponse(_error_body(exc.code, exc.message), status_code=exc.http_status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields: dict[str, str] = {}
        first = "The request contained invalid values."
        for item in exc.errors():
            loc = [str(p) for p in item.get("loc", ()) if p != "body"]
            message = str(item.get("msg", "Invalid value")).removeprefix("Value error, ")
            if item.get("type") in {"string_too_short", "string_too_long"} and loc[-1:] == ["location"]:
                message = "Enter a city or area name between 2 and 120 characters."
            fields[loc[-1] if loc else "__all__"] = message
            if first.startswith("The request"):
                first = message
        return JSONResponse(_error_body("invalid_input", first, fields=fields), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        if request.url.path.startswith("/api/") or request.url.path in {"/health", "/ready"}:
            return JSONResponse(_error_body(f"http_{exc.status_code}", message), status_code=exc.status_code,
                                headers=getattr(exc, "headers", None))
        title = "Page not found" if exc.status_code == 404 else "Something went wrong"
        return templates.TemplateResponse(request=request, name="error.html", status_code=exc.status_code,
                                          context={"active": "", "status": exc.status_code, "title": title, "message": message})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(_error_body("internal_error", "Something went wrong on our side. Please try again."), status_code=500)

    app.include_router(pages.router)
    app.include_router(api.router)
    return app
