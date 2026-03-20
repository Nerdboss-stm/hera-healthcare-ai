"""Clinical Reasoning Orchestrator — Coordinates the multi-agent pipeline.

Manages the Triage → Diagnostic → Treatment flow, passes context between
agents, records a full audit trail, and computes a consensus confidence score.
"""

from __future__ import annotations
import logging
import time

from agents.protocols import (
    PatientContext,
    ClinicalReasoningResult,
)
from agents.triage import TriageAgent
from agents.diagnostic import DiagnosticAgent
from agents.treatment import TreatmentAgent

logger = logging.getLogger(__name__)


class ClinicalOrchestrator:
    """Orchestrates the Triage → Diagnostic → Treatment agent pipeline.

    Each agent receives the outputs of prior agents, building a cumulative
    clinical reasoning chain. The orchestrator records an audit trail of
    every decision point for explainability and compliance.
    """

    def __init__(
        self,
        risk_predictor=None,
        summarizer=None,
    ):
        self.triage_agent = TriageAgent(risk_predictor=risk_predictor)
        self.diagnostic_agent = DiagnosticAgent(summarizer=summarizer)
        self.treatment_agent = TreatmentAgent()

    def reason(self, ctx: PatientContext) -> ClinicalReasoningResult:
        """Execute the full clinical reasoning pipeline."""
        start = time.time()
        audit = []

        # ── Stage 1: Triage ──────────────────────────────────────
        logger.info("Stage 1/3: Triage for patient %s", ctx.patient_id)
        triage = self.triage_agent.assess(ctx)
        audit.append({
            "stage": "triage",
            "agent": "TriageAgent",
            "esi_level": int(triage.esi_level),
            "risk_score": triage.risk_score,
            "vital_flags": triage.vital_flags,
            "rationale": triage.acuity_rationale,
        })

        # ── Stage 2: Diagnosis ───────────────────────────────────
        logger.info("Stage 2/3: Diagnosis for patient %s", ctx.patient_id)
        diagnosis = self.diagnostic_agent.diagnose(ctx, triage)
        audit.append({
            "stage": "diagnosis",
            "agent": "DiagnosticAgent",
            "primary_diagnosis": diagnosis.primary_diagnosis,
            "confidence": diagnosis.confidence,
            "differentials_count": len(diagnosis.differentials),
            "reasoning_chain": diagnosis.reasoning_chain,
        })

        # ── Stage 3: Treatment ───────────────────────────────────
        logger.info("Stage 3/3: Treatment for patient %s", ctx.patient_id)
        treatment = self.treatment_agent.plan(ctx, diagnosis)
        audit.append({
            "stage": "treatment",
            "agent": "TreatmentAgent",
            "disposition": treatment.disposition,
            "medication_count": len(treatment.medications),
            "evidence_grade": treatment.evidence_grade,
            "precautions": treatment.precautions,
        })

        # ── Consensus Score ──────────────────────────────────────
        # Weighted combination: triage risk, diagnostic confidence,
        # evidence grade strength
        grade_map = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}
        evidence_score = grade_map.get(treatment.evidence_grade, 0.5)
        consensus = round(
            0.3 * triage.risk_score
            + 0.4 * diagnosis.confidence
            + 0.3 * evidence_score,
            3,
        )

        latency_ms = round((time.time() - start) * 1000, 1)
        logger.info(
            "Pipeline complete for %s in %.1fms (consensus=%.3f)",
            ctx.patient_id, latency_ms, consensus,
        )

        return ClinicalReasoningResult(
            patient_id=ctx.patient_id,
            triage=triage,
            diagnosis=diagnosis,
            treatment=treatment,
            reasoning_audit=audit,
            consensus_score=consensus,
            pipeline_latency_ms=latency_ms,
        )
