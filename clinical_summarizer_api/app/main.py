from fastapi import FastAPI, HTTPException
from clinical_summarizer_api.app.models import NoteRequest, SummaryResponse
from clinical_summarizer_api.app.summarizer import generate_summary
from clinical_summarizer_api.app.db_logger import log_to_db
from datetime import datetime
from fastapi.responses import PlainTextResponse
from clinical_summarizer_api.app.metrics import track_metrics, prometheus_metrics

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return prometheus_metrics()

app = FastAPI(title="Clinical Summarizer API")

@app.post("/summarize", response_model=SummaryResponse)
def summarize_note(request: NoteRequest):
    try:
        summary = generate_summary(request.note)
        log_to_db(request.note, summary, "SUCCESS")
        return SummaryResponse(summary=summary, timestamp=datetime.now())
    except Exception as e:
        log_to_db(request.note, "ERROR", "FAILURE")
        raise HTTPException(status_code=500, detail=str(e))

