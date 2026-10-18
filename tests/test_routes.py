import re
import shutil

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Database
from app.services.prediction_service import PredictionService
from app.services.weather_service import WeatherService
from app.web import create_app
from tests.helpers import GEOCODE_OK, forecast_payload, mock_client

PAGES = ["/", "/dashboard", "/predict", "/how-it-works", "/model-performance", "/about"]
FOOTER = "© 2026 Dipesh Vishnu Bante. All rights reserved."


def make_app(tmp_path, *, models_dir=None, client=None, db=None, **settings):
    s = Settings(models_dir=models_dir or tmp_path / "no_models", database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                 rate_limit_per_minute=settings.pop("rate_limit_per_minute", 0), **settings)
    return create_app(s, weather=WeatherService(s, client=client or mock_client()), predictor=PredictionService(s.models_dir),
                      db=db or Database(s.database_url))


@pytest.fixture()
def client(tmp_path):
    with TestClient(make_app(tmp_path)) as c:
        yield c


def test_every_page_renders_with_shared_navigation_and_footer(client):
    for path in PAGES:
        r = client.get(path)
        assert r.status_code == 200, path
        html = r.text
        assert FOOTER in html and "not an official flood warning" in html
        for href in PAGES:
            assert f'href="{href}"' in html, (path, href)
        assert 'aria-current="page"' in html                      # active page is indicated
        assert 'type="file"' not in html and "leaflet" not in html.lower() and "<canvas" not in html   # no upload, no maps


def test_unknown_page_is_a_friendly_404_and_api_errors_are_json(client):
    r = client.get("/nope")
    assert r.status_code == 404 and "Page not found" in r.text
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_404"


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["model"]["status"] == "missing" and body["database"]["available"] is True


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff" and "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_model_performance_without_model_is_honest(client):
    html = client.get("/model-performance").text
    assert "Evaluation metrics are not available" in html and "scripts/train_model.py" in html
    assert "Confusion matrix" not in html


def test_live_prediction_success_uses_fallback_and_says_so(client):
    r = client.post("/api/predict", json={"location": "Mumbai"})
    assert r.status_code == 200
    body = r.json()
    assert body["location"]["name"] == "Mumbai" and body["data"]["status"] == "live"
    assert body["prediction"]["method"] == "weather_fallback" and body["prediction"]["fallback_reason"] == "model_missing"
    assert body["weather"]["past_72h_precipitation_mm"] == 0.0 and len(body["hourly_next"]) == 6
    # second call within the TTL is labelled as cached
    assert client.post("/api/predict", json={"location": "mumbai"}).json()["data"]["status"] == "cached"


def test_live_prediction_validation_and_errors(tmp_path):
    with TestClient(make_app(tmp_path)) as c:
        r = c.post("/api/predict", json={"location": "a"})
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"
        assert c.post("/api/predict", json={}).status_code == 422
        assert c.post("/api/predict", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
    with TestClient(make_app(tmp_path, client=mock_client(geocode={"results": []}))) as c:
        r = c.post("/api/predict", json={"location": "zzzz"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "location_not_found"
    import httpx
    with TestClient(make_app(tmp_path, client=mock_client(forecast_error=httpx.ReadTimeout("slow")))) as c:
        r = c.post("/api/predict", json={"location": "Mumbai"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "timeout"


def test_live_prediction_with_missing_fields_still_answers_and_never_invents(tmp_path):
    payload = forecast_payload(humidity=None)
    with TestClient(make_app(tmp_path, client=mock_client(forecast=payload))) as c:
        body = c.post("/api/predict", json={"location": "Mumbai"}).json()
    assert body["weather"]["humidity_pct"] is None and "humidity_pct" in body["weather"]["missing_fields"]


def test_manual_form_validation(client):
    r = client.post("/predict", data={"precip_current_mm": "-3"})
    assert r.status_code == 422 and "Must be between" in r.text
    r = client.post("/predict", data={"precip_current_mm": "abc"})
    assert r.status_code == 422 and "Enter a number." in r.text
    r = client.post("/predict", data={"humidity_pct": "50"})
    assert r.status_code == 422 and "current rainfall or the 6-hour forecast" in r.text
    r = client.post("/predict", data={"precip_current_mm": "6", "precip_past_3h_mm": "2"})
    assert r.status_code == 422 and "longer window includes the shorter one" in r.text


def test_manual_form_success_is_labelled_fallback_without_model(client):
    r = client.post("/predict", data={"precip_current_mm": "20", "precip_next_6h_mm": "30", "rain_probability_pct": "90"})
    assert r.status_code == 200
    assert "Weather-based fallback (heuristic)" in r.text and "High risk" in r.text and "not a probability" in r.text
    assert "official flood warning" in r.text


def test_manual_json_api(client):
    ok = client.post("/api/predict/manual", json={"precip_current_mm": 1.0})
    assert ok.status_code == 200 and ok.json()["prediction"]["method"] == "weather_fallback"
    assert client.post("/api/predict/manual", json={"precip_current_mm": -1}).status_code == 422
    assert client.post("/api/predict/manual", json={"precip_current_mm": 1, "bogus": 1}).status_code == 422
    assert client.post("/api/predict/manual", json={"temperature_c": 20}).status_code == 422


def test_manual_with_trained_model_requires_all_inputs_and_area(tmp_path, trained_models_dir):
    with TestClient(make_app(tmp_path, models_dir=trained_models_dir)) as c:
        full = {"precip_current_mm": "4", "precip_past_3h_mm": "9", "precip_past_24h_mm": "20", "precip_past_72h_mm": "35",
                "temperature_c": "24", "humidity_pct": "90", "wind_speed_kmh": "15"}
        used = c.post("/predict", data={**full, "model_area": "covered"})
        assert used.status_code == 200 and "Trained ML model" in used.text and "NOT calibrated" in used.text
        not_confirmed = c.post("/predict", data={**full, "model_area": "other"})
        assert "Weather-based fallback (heuristic)" in not_confirmed.text
        partial = {k: v for k, v in full.items() if k != "precip_past_72h_mm"}
        assert "Weather-based fallback (heuristic)" in c.post("/predict", data={**partial, "model_area": "covered"}).text
        perf = c.get("/model-performance").text
        assert "Confusion matrix" in perf and "Macro F1" in perf and "offline" in perf.lower()


def test_history_is_saved_and_dashboard_uses_real_records(tmp_path):
    with TestClient(make_app(tmp_path)) as c:
        assert "No predictions saved yet" in c.get("/dashboard").text
        for _ in range(5):
            c.post("/predict", data={"precip_current_mm": "1"})
        html = c.get("/dashboard").text
        assert "No predictions saved yet" not in html and "Recent predictions" in html and "5 saved predictions" in html


def test_app_works_when_the_database_is_down(tmp_path):
    bad = Database("postgresql://nobody@127.0.0.1:1/none")
    with TestClient(make_app(tmp_path, db=bad)) as c:
        assert c.get("/dashboard").status_code == 200 and "history database is unavailable" in c.get("/dashboard").text
        assert c.post("/predict", data={"precip_current_mm": "1"}).status_code == 200
        assert c.post("/api/predict", json={"location": "Mumbai"}).status_code == 200


def test_rate_limit_and_body_limit(tmp_path):
    with TestClient(make_app(tmp_path, rate_limit_per_minute=3)) as c:
        codes = [c.post("/api/predict/manual", json={"precip_current_mm": 1}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[3] == 429
    with TestClient(make_app(tmp_path, max_body_bytes=1024)) as c:
        r = c.post("/api/predict", content=b"x" * 5000, headers={"content-type": "application/json"})
        assert r.status_code == 413


def test_about_page_has_required_details_and_placeholders(client):
    html = client.get("/about").text
    assert "Dipesh Vishnu Bante" in html and "https://linkedin.com/in/dipesh-vishnu-bante-ny" in html
    assert "[Add college name]" in html and "[Add GitHub repository URL]" in html
