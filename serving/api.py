import os
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse, FileResponse
from serving.schemas import (
    NoteRequest, SummaryResponse,
    VitalsRequest, RiskResponse,
    HealthResponse,
)
from serving.summarizer import generate_summary
from serving.risk_predictor import predict_risk
from serving.db_logger import log_to_db
from serving.metrics import (
    REQUEST_COUNT, REQUEST_FAILURES, REQUEST_LATENCY,
    prometheus_metrics,
)

_dir = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="HERA — Healthcare Risk Analytics",
    description="Patient risk prediction and clinical note summarization platform",
    version="1.0.0",
)

# Serve the frontend
app.mount("/static", StaticFiles(directory=os.path.join(_dir, "static")), name="static")


# ── Health & Metrics ──────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
def serve_ui():
    return FileResponse(os.path.join(_dir, "static", "index.html"))


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    summarizer_status = "available"
    try:
        from serving.summarizer import _load_model
        _load_model()
    except Exception:
        summarizer_status = "model not loaded"

    risk_status = "available"
    try:
        from serving.risk_predictor import _load_risk_model
        _load_risk_model()
    except Exception:
        risk_status = "model not loaded"

    return HealthResponse(
        status="healthy",
        services={
            "summarizer": summarizer_status,
            "risk_predictor": risk_status,
        },
        version="1.0.0",
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return prometheus_metrics()


# ── Clinical Summarization ────────────────────────────────────────

@app.post("/api/summarize", response_model=SummaryResponse)
def summarize_note(request: NoteRequest):
    REQUEST_COUNT.inc()
    start = time.time()
    try:
        summary = generate_summary(request.note)
        log_to_db(request.note, summary, "SUCCESS")
        REQUEST_LATENCY.observe(time.time() - start)
        return SummaryResponse(
            summary=summary,
            timestamp=datetime.now(),
            note_length=len(request.note),
            summary_length=len(summary),
        )
    except Exception as e:
        REQUEST_FAILURES.inc()
        REQUEST_LATENCY.observe(time.time() - start)
        log_to_db(request.note, str(e), "FAILURE")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize/upload", response_model=SummaryResponse)
async def summarize_file(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()
    start = time.time()
    try:
        content = await file.read()
        note_text = content.decode("utf-8").strip()
        if len(note_text) < 10:
            raise HTTPException(status_code=400, detail="File content too short")
        summary = generate_summary(note_text)
        log_to_db(note_text, summary, "SUCCESS")
        REQUEST_LATENCY.observe(time.time() - start)
        return SummaryResponse(
            summary=summary,
            timestamp=datetime.now(),
            note_length=len(note_text),
            summary_length=len(summary),
        )
    except UnicodeDecodeError:
        REQUEST_FAILURES.inc()
        raise HTTPException(status_code=400, detail="File must be a UTF-8 text file")
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_FAILURES.inc()
        REQUEST_LATENCY.observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── Risk Prediction ──────────────────────────────────────────────

@app.post("/api/predict", response_model=RiskResponse)
def predict_patient_risk(request: VitalsRequest):
    REQUEST_COUNT.inc()
    start = time.time()
    try:
        result = predict_risk(
            heart_rate=request.heart_rate,
            respiratory_rate=request.respiratory_rate,
            body_temperature=request.body_temperature,
            oxygen_saturation=request.oxygen_saturation,
            systolic_bp=request.systolic_bp,
            diastolic_bp=request.diastolic_bp,
            age=request.age,
        )
        REQUEST_LATENCY.observe(time.time() - start)
        return RiskResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            features_used=result["features_used"],
            timestamp=datetime.now(),
        )
    except Exception as e:
        REQUEST_FAILURES.inc()
        REQUEST_LATENCY.observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))
