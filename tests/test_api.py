"""Tests of the HTTP API (backend/main.py). Skipped when the models or templates are missing."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ENABLED_MODELS", "classical")

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend import main  # noqa: E402

pytestmark = pytest.mark.skipif(not main.MODELS or not main.library().templates,
                                reason="trained model / templates not available")
client = TestClient(main.app)


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_health_lists_models_and_layouts():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "classical" in body["models"]
    assert {"axis", "icici", "canara", "syndicate"} <= set(body["templates"])


def test_models_endpoint_has_metrics_and_one_recommendation():
    body = client.get("/api/models").json()
    assert sum(m["recommended"] for m in body["models"]) == 1
    classical = next(m for m in body["models"] if m["id"] == "classical")
    assert 0.5 < classical["metrics"]["cheque_auc"] <= 1.0
    assert {f["id"] for f in body["fields"]} == {"payee", "words", "figures", "date", "signature"}


def test_samples_are_listed_and_downloadable():
    samples = client.get("/api/samples").json()
    if not samples:
        pytest.skip("no demo cheques bundled")
    r = client.get("/api/samples/" + samples[0]["name"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("bad", ["../config.py", "..%2Fconfig.py", "nope.jpg"])
def test_sample_names_are_whitelisted(bad):
    assert client.get("/api/samples/" + bad).status_code == 404


def test_detect_demo_cheques_return_expected_verdicts():
    samples = {s["name"]: s for s in client.get("/api/samples").json()}
    fraud = next((n for n, s in samples.items() if s["tampered"]), None)
    genuine = next((n for n, s in samples.items() if not s["tampered"]), None)
    if not (fraud and genuine):
        pytest.skip("no demo cheques bundled")
    f = client.post("/api/detect", data={"sample": fraud, "model": "classical"})
    assert f.status_code == 200
    body = f.json()
    assert body["verdict"] == "FRAUD" and body["units_suspicious"] > 0
    assert body["mode"] == "template" and body["images"]["highlighted"].startswith("data:image/jpeg;base64,")
    assert 0.5 <= body["confidence"] <= 1.0
    g = client.post("/api/detect", data={"sample": genuine, "model": "classical"}).json()
    assert g["verdict"] == "GENUINE" and g["units_suspicious"] == 0


def test_upload_and_threshold_override():
    samples = client.get("/api/samples").json()
    if not samples:
        pytest.skip("no demo cheques bundled")
    data = open(os.path.join(main.SAMPLE_DIR, samples[0]["name"]), "rb").read()
    strict = client.post("/api/detect", files={"file": ("c.jpg", data, "image/jpeg")},
                         data={"model": "classical", "threshold": "0.99"}).json()
    assert strict["threshold"] == 0.99 and strict["verdict"] == "GENUINE"


def test_input_validation():
    assert client.post("/api/detect", data={}).status_code == 400                       # nothing sent
    assert client.post("/api/detect", data={"sample": "x.jpg"}).status_code == 404
    assert client.post("/api/detect", data={"sample": "a", "model": "bogus"}).status_code == 400
    assert client.post("/api/detect", files={"file": ("a.txt", b"not an image", "text/plain")},
                       data={"model": "classical"}).status_code == 400
    assert client.post("/api/detect", files={"file": ("a.png", b"", "image/png")},
                       data={"model": "classical"}).status_code == 400
    ok = _png(np.full((50, 50, 3), 255, np.uint8))
    assert client.post("/api/detect", files={"file": ("a.png", ok, "image/png")},
                       data={"model": "classical", "threshold": "5"}).status_code == 400


def test_blank_image_is_handled_without_crashing():
    blank = _png(np.full((1100, 2365, 3), 235, np.uint8))
    r = client.post("/api/detect", files={"file": ("blank.png", blank, "image/png")},
                    data={"model": "classical"})
    assert r.status_code == 200
    assert r.json()["verdict"] in ("INSUFFICIENT INK", "GENUINE")


def test_oversized_image_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "MAX_PIXELS", 1000)
    img = _png(np.full((100, 100, 3), 200, np.uint8))
    r = client.post("/api/detect", files={"file": ("big.png", img, "image/png")}, data={"model": "classical"})
    assert r.status_code == 413


def test_cors_allows_configured_origin():
    r = client.options("/api/detect", headers={"Origin": "http://localhost:5173",
                                               "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r = client.options("/api/detect", headers={"Origin": "https://evil.example",
                                               "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers
