from fastapi import FastAPI, HTTPException
from serving.schemas import NoteRequest, SummaryResponse
from serving.summarizer import generate_summary
from serving.db_logger import log_to_db
from datetime import datetime
from fastapi.responses import PlainTextResponse
from serving.metrics import track_metrics, prometheus_metrics

def metrics():
    return prometheus_metrics()

app = FastAPI(title="Clinical Summarizer API")

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return prometheus_metrics()

@app.get("/")
def root():
    return {"message": "FastAPI server is up!"}

@app.post("/summarize", response_model=SummaryResponse)
def summarize_note(request: NoteRequest):
    try:
        summary = generate_summary(request.note)
        log_to_db(request.note, summary, "SUCCESS")
        return SummaryResponse(summary=summary, timestamp=datetime.now())
    except Exception as e:
        log_to_db(request.note, "ERROR", "FAILURE")
        raise HTTPException(status_code=500, detail=str(e))

