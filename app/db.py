"""Prediction history (SQLAlchemy). SQLite for local development, PostgreSQL for production.

Initialisation strategy: ``Database.init_schema()`` runs ``CREATE TABLE IF NOT EXISTS`` at start-up, so a fresh
deployment never fails on a missing table. There are no migrations yet: if you change a model, add Alembic or
migrate by hand (see README, "Database"). Only weather numbers, a coarse place name and rounded coordinates are
stored - no IP addresses, user agents or other personal data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

log = logging.getLogger("waterlogging.db")


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(16))                       # "live" | "manual"
    location_name: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    risk_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    method: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    fallback_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    data_timestamp: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    inputs: Mapped[dict] = mapped_column(JSON)


class Database:
    def __init__(self, url: str):
        self.url = url
        self.available = False
        self.error: Optional[str] = None
        self._engine = None
        self._sessions = None
        try:
            parsed = make_url(url)
            kwargs: dict[str, Any] = {"pool_pre_ping": True}
            if parsed.get_backend_name() == "sqlite":
                if parsed.database and parsed.database != ":memory:":
                    Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
                kwargs["connect_args"] = {"check_same_thread": False}
            self._engine = create_engine(url, **kwargs)
            self._sessions = sessionmaker(self._engine, expire_on_commit=False)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("database engine could not be created: %s", self.error)

    def init_schema(self) -> bool:
        if self._engine is None:
            return False
        try:
            Base.metadata.create_all(self._engine)
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.available, self.error = True, None
        except Exception as exc:
            self.available, self.error = False, f"{type(exc).__name__}"
            log.error("database unavailable; history disabled (%s)", type(exc).__name__)
        return self.available

    def add_prediction(self, *, source: str, result: dict[str, Any], inputs: dict[str, Any],
                       location_name: Optional[str] = None, latitude: Optional[float] = None,
                       longitude: Optional[float] = None) -> bool:
        if not self.available:
            return False
        record = PredictionRecord(
            created_at=datetime.now(timezone.utc), source=source, location_name=(location_name or None) and location_name[:160],
            latitude=round(latitude, 2) if latitude is not None else None,
            longitude=round(longitude, 2) if longitude is not None else None,
            risk_level=result["risk_level"], risk_index=result.get("risk_index"), method=result["method"],
            model_version=result.get("model_version"), fallback_reason=result.get("fallback_reason"),
            data_timestamp=result.get("data_timestamp"), inputs={k: v for k, v in inputs.items() if v is None or isinstance(v, (int, float))},
        )
        try:
            with self._sessions() as session:
                session.add(record)
                session.commit()
            return True
        except Exception as exc:
            log.warning("could not save prediction (%s)", type(exc).__name__)
            return False

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            with self._sessions() as session:
                rows = session.scalars(select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit)).all()
                return [{"created_at": r.created_at, "source": r.source, "location_name": r.location_name,
                         "risk_level": r.risk_level, "risk_index": r.risk_index, "method": r.method,
                         "model_version": r.model_version, "fallback_reason": r.fallback_reason} for r in rows]
        except Exception as exc:
            log.warning("history query failed (%s)", type(exc).__name__)
            return []

    def stats(self) -> dict[str, Any]:
        """Totals and distributions over genuinely saved predictions."""
        empty = {"total": 0, "by_risk": {}, "by_method": {}}
        if not self.available:
            return empty
        try:
            with self._sessions() as session:
                total = session.scalar(select(func.count(PredictionRecord.id))) or 0
                by_risk = dict(session.execute(select(PredictionRecord.risk_level, func.count()).group_by(PredictionRecord.risk_level)).all())
                by_method = dict(session.execute(select(PredictionRecord.method, func.count()).group_by(PredictionRecord.method)).all())
            return {"total": int(total), "by_risk": {k: int(v) for k, v in by_risk.items()},
                    "by_method": {k: int(v) for k, v in by_method.items()}}
        except Exception as exc:
            log.warning("history stats failed (%s)", type(exc).__name__)
            return empty

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
