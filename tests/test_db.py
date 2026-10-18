from app.db import Database


def _result(level="Low", method="weather_fallback"):
    return {"risk_level": level, "risk_index": 12, "method": method, "model_version": None, "fallback_reason": "model_missing",
            "data_timestamp": "2026-10-17T14:15"}


def test_fresh_database_initialises_and_stores(tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'sub' / 'w.db').as_posix()}")     # directory does not exist yet
    assert db.init_schema() and db.available
    assert db.stats() == {"total": 0, "by_risk": {}, "by_method": {}} and db.recent() == []
    assert db.add_prediction(source="live", result=_result("High"), inputs={"precip_current_mm": 3.0, "x": "ignored"},
                             location_name="Mumbai", latitude=19.0761234, longitude=72.8777)
    assert db.add_prediction(source="manual", result=_result("Low"), inputs={})
    stats = db.stats()
    assert stats["total"] == 2 and stats["by_risk"] == {"High": 1, "Low": 1}
    rows = db.recent(5)
    assert rows[0]["source"] == "manual" and rows[1]["location_name"] == "Mumbai"


def test_init_is_idempotent(tmp_path):
    url = f"sqlite:///{(tmp_path / 'w.db').as_posix()}"
    assert Database(url).init_schema() and Database(url).init_schema()


def test_unavailable_database_never_raises():
    db = Database("postgresql://nobody@127.0.0.1:1/none")      # no server (and possibly no driver): must degrade quietly
    assert db.init_schema() is False and db.available is False
    assert db.add_prediction(source="live", result=_result(), inputs={}) is False
    assert db.recent() == [] and db.stats()["total"] == 0
