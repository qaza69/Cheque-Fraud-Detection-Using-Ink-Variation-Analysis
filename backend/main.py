"""HTTP API of the cheque tamper detector (FastAPI).

    GET  /health               liveness + which models are loaded
    GET  /api/models           available models with their cross-validated metrics
    GET  /api/samples          demo cheques bundled with the service
    GET  /api/samples/{name}   one demo cheque image
    POST /api/detect           analyse a cheque (uploaded file or demo name)

Privacy: uploaded cheques are processed in memory only - nothing is written to disk,
logged or kept after the response is sent.

Run locally:   uvicorn backend.main:app --reload --port 8000
"""
import base64
import os
import sys
import threading
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "backend", "artifacts")

# Use the artefacts shipped with the service when present (templates, models, samples);
# otherwise fall back to the folders used during development.
if os.path.isdir(os.path.join(ARTIFACTS, "templates")):
    os.environ.setdefault("CHEQUE_TEMPLATE_DIR", os.path.join(ARTIFACTS, "templates"))
if os.path.isdir(os.path.join(ARTIFACTS, "models")):
    os.environ.setdefault("CHEQUE_MODEL_DIR", os.path.join(ARTIFACTS, "models"))
SAMPLE_DIR = (os.path.join(ARTIFACTS, "samples") if os.path.isdir(os.path.join(ARTIFACTS, "samples"))
              else os.path.join(ROOT, "data", "samples"))

sys.path.insert(0, ROOT)
import config  # noqa: E402
from src.fraud_pipeline import FraudDetector  # noqa: E402
from src.ink.extraction import DEFAULT_FIELDS, FIELD_LABELS  # noqa: E402
from src.ink.registry import MODEL_LABELS, available_models, load_metrics  # noqa: E402
from src.ink.templates import BANK_NAMES, TemplateLibrary  # noqa: E402

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "12")) * 1024 * 1024
MAX_PIXELS = int(os.environ.get("MAX_MEGAPIXELS", "40")) * 1_000_000
RESPONSE_WIDTH = 1200                       # width of the images sent back to the browser
ALLOWED_FIELDS = tuple(FIELD_LABELS.keys() - {"all"})

enabled = [m.strip() for m in os.environ.get("ENABLED_MODELS", "").split(",") if m.strip()]
MODELS = [m for m in available_models() if not enabled or m in enabled]

app = FastAPI(title="Cheque Fraud Detection API", version="1.0.0",
              description="Detects cheques altered with a second pen by comparing the ink of "
                          "the handwritten fields.")

origins = [o.strip() for o in os.environ.get(
    "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins,
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX") or None,   # e.g. Vercel preview URLs
    allow_methods=["GET", "POST"], allow_headers=["*"], max_age=600)

_library = None
_detectors = {}
_lock = threading.RLock()                   # guards lazy loading
_busy = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT", "1")))   # RAM guard


def library():
    global _library
    with _lock:
        if _library is None:
            _library = TemplateLibrary()
        return _library


def detector(name):
    with _lock:
        if name not in _detectors:
            _detectors[name] = FraudDetector(model_type=name, library=library())
        return _detectors[name]


def _metrics_summary(name):
    m = load_metrics().get(name, {})
    keep = ("cheque_auc", "accuracy", "precision", "recall", "f1", "specificity", "pair_auc",
            "threshold", "full_cheque_detection", "n_cheques")
    return {k: m[k] for k in keep if k in m}


def _to_jpeg_b64(rgb, width=RESPONSE_WIDTH, quality=82):
    h, w = rgb.shape[:2]
    if w > width:
        rgb = cv2.resize(rgb, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "Could not encode the result image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "The file is not a readable image (use JPG, PNG, BMP or TIFF).")
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        raise HTTPException(413, f"Image is too large ({w}x{h}). Please scale it down.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _sample_files():
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def _sample_label(fname):
    parts = os.path.splitext(fname)[0].split("_", 2)
    if len(parts) != 3:
        return fname
    kind, bank, cid = parts
    return f"{'Tampered' if kind == 'fraud' else 'Genuine'} - {BANK_NAMES.get(bank, bank)} #{cid}"


# --------------------------------------------------------------------- routes
@app.get("/")
def root():
    return {"service": "Cheque Fraud Detection API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "models": MODELS, "templates": sorted(library().templates)}


@app.get("/api/models")
def list_models():
    recommended = next((m for m in MODELS if load_metrics().get(m, {}).get("recommended")),
                       MODELS[0] if MODELS else None)
    return {
        "models": [{"id": m, "label": MODEL_LABELS[m], "recommended": m == recommended,
                    "metrics": _metrics_summary(m)} for m in MODELS],
        "fields": [{"id": f, "label": FIELD_LABELS[f], "default": f in DEFAULT_FIELDS}
                   for f in ("payee", "words", "figures", "date", "signature")],
        "layouts": [BANK_NAMES[k] for k in sorted(library().templates)],
    }


@app.get("/api/samples")
def list_samples():
    return [{"name": f, "label": _sample_label(f), "tampered": f.startswith("fraud")}
            for f in _sample_files()]


@app.get("/api/samples/{name}")
def get_sample(name: str):
    if name not in _sample_files():                      # whitelist: no path tricks
        raise HTTPException(404, "Unknown sample")
    return FileResponse(os.path.join(SAMPLE_DIR, name), media_type="image/jpeg")


@app.post("/api/detect")
async def detect(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Form(None),
    model: str = Form(""),
    threshold: Optional[float] = Form(None),
    fields: List[str] = Form(default=[]),
):
    if not MODELS:
        raise HTTPException(503, "No trained model is available on the server")
    model = model or (MODELS[0])
    if model not in MODELS:
        raise HTTPException(400, f"Model must be one of {MODELS}")
    chosen = tuple(f for f in fields if f in ALLOWED_FIELDS) or DEFAULT_FIELDS
    if threshold is not None and not 0.01 <= threshold <= 0.99:
        raise HTTPException(400, "threshold must be between 0.01 and 0.99")

    if file is not None:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(400, "Empty file")
    elif sample:
        if sample not in _sample_files():
            raise HTTPException(404, "Unknown sample")
        with open(os.path.join(SAMPLE_DIR, sample), "rb") as fh:
            data = fh.read()
    else:
        raise HTTPException(400, "Send a cheque image as 'file' or a demo name as 'sample'")

    image = _decode_image(data)
    del data

    if not _busy.acquire(timeout=90):
        raise HTTPException(503, "The server is busy, please try again in a moment")
    try:
        result = _run(model, image, threshold, chosen)
    finally:
        _busy.release()
    return result


def _run(model, image, threshold, fields):
    try:
        res = detector(model).detect(image, rgb=True, threshold=threshold, fields=fields)
    except Exception as err:                                   # pipeline failure, not user error
        raise HTTPException(500, f"Detection failed: {err}") from err

    units = res["units"]
    P = np.asarray(res["pair_matrix"], dtype=float)
    order = np.argsort([u["distinct"] for u in units]).tolist() if len(units) else []
    matrix = np.round(P[np.ix_(order, order)], 2).tolist() if len(order) else []
    h, w = res["highlighted_image"].shape[:2]
    return {
        "verdict": res["verdict"],
        "confidence": round(float(res["confidence"]), 4),
        "score": round(float(res["score"]), 4),
        "threshold": round(float(res["threshold"]), 4),
        "model": model,
        "model_label": res["model"],
        "layout": res["bank"],
        "mode": res["mode"],
        "match_correlation": round(float(res["match_cc"]), 3),
        "units_total": len(units),
        "units_suspicious": int(sum(u["suspicious"] for u in units)),
        "field_summary": {k: {"units": int(v["units"]), "suspicious": int(v["suspicious"]),
                              "max_distinct": round(float(v["max_distinct"]), 3)}
                          for k, v in res["field_summary"].items()},
        "notes": res["notes"],
        "image_size": {"width": int(w), "height": int(h)},
        "units": [{"field": u["field"], "x": u["x"], "y": u["y"], "w": u["w"], "h": u["h"],
                   "distinct": round(float(u["distinct"]), 3), "suspicious": bool(u["suspicious"])}
                  for u in units],
        "pair_matrix_sorted": matrix,
        "images": {
            "highlighted": _to_jpeg_b64(res["highlighted_image"]),
            "ink": _to_jpeg_b64(res["ink_image"]),
            "heatmap": _to_jpeg_b64(res["heatmap"]),
        },
    }
