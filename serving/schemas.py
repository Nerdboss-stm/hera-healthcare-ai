from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict


class NoteRequest(BaseModel):
    note: str = Field(..., min_length=10, description="Clinical note text to summarize")


class SummaryResponse(BaseModel):
    summary: str
    timestamp: datetime
    note_length: Optional[int] = None
    summary_length: Optional[int] = None


class VitalsRequest(BaseModel):
    heart_rate: float = Field(..., ge=30, le=220, description="Heart rate in BPM")
    respiratory_rate: float = Field(..., ge=5, le=60, description="Breaths per minute")
    body_temperature: float = Field(..., ge=34.0, le=42.0, description="Temperature in Celsius")
    oxygen_saturation: float = Field(..., ge=70, le=100, description="SpO2 percentage")
    systolic_bp: float = Field(..., ge=60, le=250, description="Systolic blood pressure mmHg")
    diastolic_bp: float = Field(..., ge=30, le=150, description="Diastolic blood pressure mmHg")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")


class RiskResponse(BaseModel):
    prediction: str
    confidence: float
    risk_score: float
    features_used: Dict[str, float]
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    version: str
