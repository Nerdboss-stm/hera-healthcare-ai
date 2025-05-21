from pydantic import BaseModel
from datetime import datetime

class NoteRequest(BaseModel):
    note: str

class SummaryResponse(BaseModel):
    summary: str
    timestamp: datetime

