from fastapi import FastAPI, HTTPException
from clinical_summarizer_api.app.models import NoteRequest, SummaryResponse
from clinical_summarizer_api.app.summarizer import generate_summary
from clinical_summarizer_api.app.db_logger import log_to_db
from datetime import datetime

app = FastAPI(title="Clinical Summarizer API")

@app.get("/")
def root():
    return {"message": "Welcome to the Clinical Summarizer API. Use /docs to explore endpoints."}

@app.post("/summarize", response_model=SummaryResponse)
def summarize_note(request: NoteRequest):
    try:
        summary = generate_summary(request.note)
        log_to_db(request.note, summary, "SUCCESS")
        return SummaryResponse(summary=summary, timestamp=datetime.now())
    except Exception as e:
        log_to_db(request.note, "ERROR", "FAILURE")
        raise HTTPException(status_code=500, detail=str(e))

