"""Runtime configuration, read once from environment variables (see ``.env.example``)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_VERSION = "2.0.0"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_number(name: str, default: float, cast=float, minimum: float | None = None):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


@dataclass(frozen=True)
class Settings:
    env: str = "development"
    database_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'waterlogging.db').as_posix()}"
    models_dir: Path = BASE_DIR / "models"
    http_timeout_s: float = 10.0
    weather_cache_ttl_s: int = 600          # a cache hit younger than this is served and labelled "cached"
    weather_stale_ttl_s: int = 10800        # if the provider is down, data up to this old is served labelled "stale"
    rate_limit_per_minute: int = 30         # per client address, per process; 0 disables
    max_body_bytes: int = 16_384
    save_history: bool = True
    log_level: str = "INFO"
    geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


def load_settings() -> Settings:
    d = Settings()
    db_url = os.getenv("WW_DATABASE_URL", d.database_url).strip() or d.database_url
    if db_url.startswith("postgres://"):          # some hosts still hand out the legacy scheme
        db_url = "postgresql://" + db_url[len("postgres://"):]
    return Settings(
        env=os.getenv("WW_ENV", d.env),
        database_url=db_url,
        models_dir=Path(os.getenv("WW_MODELS_DIR", str(d.models_dir))),
        http_timeout_s=_env_number("WW_HTTP_TIMEOUT_S", d.http_timeout_s, float, 1),
        weather_cache_ttl_s=_env_number("WW_WEATHER_CACHE_TTL_S", d.weather_cache_ttl_s, int, 0),
        weather_stale_ttl_s=_env_number("WW_WEATHER_STALE_TTL_S", d.weather_stale_ttl_s, int, 0),
        rate_limit_per_minute=_env_number("WW_RATE_LIMIT_PER_MINUTE", d.rate_limit_per_minute, int, 0),
        max_body_bytes=_env_number("WW_MAX_BODY_BYTES", d.max_body_bytes, int, 1024),
        save_history=_env_bool("WW_SAVE_HISTORY", d.save_history),
        log_level=os.getenv("WW_LOG_LEVEL", d.log_level).upper(),
        geocoding_url=os.getenv("WW_GEOCODING_URL", d.geocoding_url),
        forecast_url=os.getenv("WW_FORECAST_URL", d.forecast_url),
    )
