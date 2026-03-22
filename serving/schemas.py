from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, List, Any


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
    body_temperature: float = Field(
        ..., ge=34.0, le=42.0, description="Temperature in Celsius"
    )
    oxygen_saturation: float = Field(..., ge=70, le=100, description="SpO2 percentage")
    systolic_bp: float = Field(
        ..., ge=60, le=250, description="Systolic blood pressure mmHg"
    )
    diastolic_bp: float = Field(
        ..., ge=30, le=150, description="Diastolic blood pressure mmHg"
    )
    age: int = Field(..., ge=0, le=120, description="Patient age in years")


class RiskResponse(BaseModel):
    prediction: str
    confidence: float
    risk_score: float
    risk_level: Optional[str] = None
    ml_binary_label: Optional[str] = None
    ml_probability: Optional[float] = None
    features_used: Dict[str, float]
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    version: str


# ── Multi-Agent Clinical Reasoning ───────────────────────────


class ClinicalReasoningRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    chief_complaint: str = Field(..., min_length=3, description="Primary complaint")
    clinical_note: str = Field(..., min_length=10, description="Full clinical note")
    heart_rate: float = Field(..., ge=30, le=220)
    respiratory_rate: float = Field(..., ge=5, le=60)
    body_temperature: float = Field(..., ge=34.0, le=42.0)
    oxygen_saturation: float = Field(..., ge=70, le=100)
    systolic_bp: float = Field(..., ge=60, le=250)
    diastolic_bp: float = Field(..., ge=30, le=150)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(default="unknown")
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)


class ClinicalReasoningResponse(BaseModel):
    patient_id: str
    triage: Dict[str, Any]
    diagnosis: Dict[str, Any]
    treatment: Dict[str, Any]
    reasoning_audit: List[Dict[str, Any]]
    consensus_score: float
    pipeline_latency_ms: float
    timestamp: datetime


# ── RAG Knowledge Base ───────────────────────────────────────


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Medical knowledge query")
    top_k: int = Field(default=3, ge=1, le=10)


class RAGQueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_corpus_size: int


class RAGSummarizeRequest(BaseModel):
    note: str = Field(
        ..., min_length=10, description="Clinical note for RAG-augmented summary"
    )
    top_k: int = Field(default=3, ge=1, le=5)


class RAGSummarizeResponse(BaseModel):
    summary: Optional[str]
    citations: List[Dict[str, Any]]
    retrieved_context: List[Dict[str, Any]]
    augmented: bool
    timestamp: datetime


# ── Clinical NER ─────────────────────────────────────────────


class NERRequest(BaseModel):
    note: str = Field(
        ..., min_length=10, description="Clinical note for entity extraction"
    )


class NERResponse(BaseModel):
    entity_count: int
    medications: List[Dict[str, str]]
    conditions: List[Dict[str, str]]
    procedures: List[Dict[str, str]]
    lab_values: List[Dict[str, str]]
    vital_signs: List[Dict[str, str]]
    timestamp: datetime


class KnowledgeGraphRequest(BaseModel):
    note: str = Field(
        ..., min_length=10, description="Clinical note for graph building"
    )
    patient_id: str = Field(default="unknown")


class KnowledgeGraphResponse(BaseModel):
    patient_id: str
    nodes: int
    edges: int
    entities: Dict[str, Any]
    graph: Dict[str, Any]
    timestamp: datetime


# ── FHIR R4 ─────────────────────────────────────────────────


class FHIRBundleRequest(BaseModel):
    bundle: Dict[str, Any] = Field(..., description="FHIR R4 Bundle resource")


class FHIRPredictResponse(BaseModel):
    risk_assessment: Dict[str, Any]
    original_prediction: Dict[str, Any]
    timestamp: datetime


# ── Evaluation ───────────────────────────────────────────────


class EvaluationRequest(BaseModel):
    source_note: str = Field(..., min_length=10)
    generated_output: str = Field(..., min_length=10)


class EvaluationResponse(BaseModel):
    overall_score: float
    pass_threshold: bool
    factual_consistency: Dict[str, Any]
    hallucination: Dict[str, Any]
    medical_accuracy: Dict[str, Any]
    clinical_safety: Dict[str, Any]
    timestamp: datetime


# ── Unified Command Center ────────────────────────────────────


class CommandCenterRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    chief_complaint: str = Field(..., min_length=3)
    clinical_note: str = Field(..., min_length=10)
    heart_rate: float = Field(..., ge=30, le=220)
    respiratory_rate: float = Field(..., ge=5, le=60)
    body_temperature: float = Field(..., ge=34.0, le=42.0)
    oxygen_saturation: float = Field(..., ge=70, le=100)
    systolic_bp: float = Field(..., ge=60, le=250)
    diastolic_bp: float = Field(..., ge=30, le=150)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(default="unknown")
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)


class CommandCenterResponse(BaseModel):
    patient_id: str
    stages: List[Dict[str, Any]]
    overall_latency_ms: float
    consensus_score: float
    safety_passed: bool
    feedback_loops_triggered: int
    fhir_bundle: Dict[str, Any]
    data_lineage: List[Dict[str, Any]]
    timestamp: datetime


# ── Usage & Admin ──────────────────────────────────────────────


class UsageResponse(BaseModel):
    tenants: Dict[str, Any]
    timestamp: datetime


# ── Data Engineering ──────────────────────────────────────────


class DEStreamingResponse(BaseModel):
    patient_id: str
    events: List[Dict[str, Any]]
    dlq: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    topic_stats: Dict[str, Any]
    schema_evolution: List[Dict[str, Any]]


class DELineageResponse(BaseModel):
    dag: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    pii_columns: List[Dict[str, Any]]


class DEQualityResponse(BaseModel):
    vitals_quality: Dict[str, Any]
    note_quality: Dict[str, Any]
    pipeline_summary: Dict[str, Any]


class DEWarehouseResponse(BaseModel):
    encounter_id: int
    warehouse_stats: Dict[str, Any]
    risk_distribution: Dict[str, Any]
    recent_encounters: List[Dict[str, Any]]


class DEOrchestratorResponse(BaseModel):
    dag_definition: Dict[str, Any]
    run_history: List[Dict[str, Any]]


class DECDCResponse(BaseModel):
    events: List[Dict[str, Any]]
    stats: Dict[str, Any]


class DECatalogResponse(BaseModel):
    catalog: Dict[str, Any]
    pii_report: List[Dict[str, Any]]
    freshness_report: List[Dict[str, Any]]


class DEDashboardResponse(BaseModel):
    streaming: Dict[str, Any]
    quality: Dict[str, Any]
    lineage: Dict[str, Any]
    warehouse: Dict[str, Any]
    orchestrator: Dict[str, Any]
    cdc: Dict[str, Any]
    catalog: Dict[str, Any]
    timestamp: datetime
