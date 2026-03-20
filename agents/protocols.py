"""Structured protocols for inter-agent communication.

Defines typed message contracts so agents exchange structured JSON,
enabling full audit trails and reasoning chain explainability.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import IntEnum
from typing import Optional


class ESILevel(IntEnum):
    """Emergency Severity Index — 5-level triage acuity scale."""
    RESUSCITATION = 1   # Immediate life-saving intervention
    EMERGENT = 2        # High risk / confused / severe pain
    URGENT = 3          # Stable but needs multiple resources
    LESS_URGENT = 4     # Stable, single resource expected
    NON_URGENT = 5      # No resources expected


@dataclass
class PatientContext:
    """Input context for the clinical reasoning pipeline."""
    patient_id: str
    chief_complaint: str
    clinical_note: str
    vitals: dict
    age: int
    gender: str = "unknown"
    medical_history: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TriageResult:
    """Output from the Triage Agent."""
    esi_level: ESILevel
    acuity_rationale: str
    vital_flags: list[str]
    immediate_actions: list[str]
    risk_score: float  # 0.0 (low) to 1.0 (critical)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["esi_level"] = int(self.esi_level)
        return d


@dataclass
class Diagnosis:
    """A single differential diagnosis entry."""
    condition: str
    icd10_code: str
    probability: float  # 0.0 to 1.0
    supporting_evidence: list[str]
    ruling_out: list[str]


@dataclass
class DiagnosticResult:
    """Output from the Diagnostic Agent."""
    differentials: list[Diagnosis]
    primary_diagnosis: str
    confidence: float
    reasoning_chain: list[str]
    recommended_tests: list[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Medication:
    """A single medication recommendation."""
    name: str
    dose: str
    route: str
    frequency: str
    rationale: str
    contraindication_check: str


@dataclass
class TreatmentResult:
    """Output from the Treatment Agent."""
    treatment_plan: list[str]
    medications: list[Medication]
    monitoring_plan: list[str]
    disposition: str  # admit / discharge / observe
    follow_up: str
    precautions: list[str]
    evidence_grade: str  # A, B, C, D
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClinicalReasoningResult:
    """Complete output from the multi-agent pipeline."""
    patient_id: str
    triage: TriageResult
    diagnosis: DiagnosticResult
    treatment: TreatmentResult
    reasoning_audit: list[dict]
    consensus_score: float
    pipeline_latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "triage": self.triage.to_dict(),
            "diagnosis": self.diagnosis.to_dict(),
            "treatment": asdict(self.treatment),
            "reasoning_audit": self.reasoning_audit,
            "consensus_score": self.consensus_score,
            "pipeline_latency_ms": self.pipeline_latency_ms,
            "timestamp": self.timestamp,
        }
