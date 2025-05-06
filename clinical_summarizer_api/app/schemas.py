from pydantic import BaseModel

class SummarizationRequest(BaseModel):
    text: str

class SummarizationResponse(BaseModel):
    summary: str

