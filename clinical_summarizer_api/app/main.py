from fastapi import FastAPI, Depends
from app.model import generate_summary
from app.schemas import SummarizationRequest, SummarizationResponse
from app.db import SessionLocal, PredictionLog
from app.logger import logger

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest, db=Depends(get_db)):
    logger.info("Received summarization request")
    
    summary = generate_summary(request.text)
    
    log_entry = PredictionLog(
        input_text=request.text,
        prediction=summary
    )
    db.add(log_entry)
    db.commit()
    
    return SummarizationResponse(summary=summary)

@app.get("/")
def root():
    return {"message": "Clinical Summarization API Running"}

