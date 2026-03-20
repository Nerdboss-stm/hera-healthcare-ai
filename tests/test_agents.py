"""Tests for the Multi-Agent Clinical Reasoning Pipeline."""

import pytest
from agents.protocols import PatientContext, ESILevel
from agents.triage import TriageAgent
from agents.diagnostic import DiagnosticAgent
from agents.treatment import TreatmentAgent
from agents.orchestrator import ClinicalOrchestrator


def _make_context(**overrides):
    defaults = dict(
        patient_id="test-001",
        chief_complaint="chest pain",
        clinical_note="65yo male with acute chest pain radiating to left arm, diaphoresis, tachycardia",
        vitals={
            "heart_rate": 110,
            "respiratory_rate": 22,
            "body_temperature": 37.2,
            "oxygen_saturation": 93,
            "systolic_bp": 90,
            "diastolic_bp": 60,
        },
        age=65,
        gender="male",
    )
    defaults.update(overrides)
    return PatientContext(**defaults)


# ── Triage Agent ─────────────────────────────────────────────

class TestTriageAgent:
    def test_chest_pain_high_acuity(self):
        agent = TriageAgent()
        ctx = _make_context()
        result = agent.assess(ctx)
        assert result.esi_level <= ESILevel.EMERGENT

    def test_low_acuity_complaint(self):
        agent = TriageAgent()
        ctx = _make_context(
            chief_complaint="sore throat",
            vitals={"heart_rate": 75, "respiratory_rate": 16,
                    "body_temperature": 37.0, "oxygen_saturation": 99,
                    "systolic_bp": 120, "diastolic_bp": 80},
        )
        result = agent.assess(ctx)
        assert result.esi_level >= ESILevel.LESS_URGENT

    def test_critical_vitals_escalate(self):
        agent = TriageAgent()
        ctx = _make_context(
            chief_complaint="weakness",
            vitals={"heart_rate": 35, "respiratory_rate": 6,
                    "body_temperature": 34.5, "oxygen_saturation": 85,
                    "systolic_bp": 70, "diastolic_bp": 35},
        )
        result = agent.assess(ctx)
        assert result.esi_level == ESILevel.RESUSCITATION

    def test_vital_flags_populated(self):
        agent = TriageAgent()
        ctx = _make_context()
        result = agent.assess(ctx)
        assert len(result.vital_flags) > 0

    def test_risk_score_range(self):
        agent = TriageAgent()
        ctx = _make_context()
        result = agent.assess(ctx)
        assert 0.0 <= result.risk_score <= 1.0

    def test_immediate_actions_for_esi1(self):
        agent = TriageAgent()
        ctx = _make_context(chief_complaint="cardiac arrest")
        result = agent.assess(ctx)
        assert len(result.immediate_actions) > 0


# ── Diagnostic Agent ─────────────────────────────────────────

class TestDiagnosticAgent:
    def test_generates_differentials(self):
        agent = DiagnosticAgent()
        ctx = _make_context()
        triage = TriageAgent().assess(ctx)
        result = agent.diagnose(ctx, triage)
        assert len(result.differentials) > 0

    def test_primary_diagnosis_set(self):
        agent = DiagnosticAgent()
        ctx = _make_context()
        triage = TriageAgent().assess(ctx)
        result = agent.diagnose(ctx, triage)
        assert result.primary_diagnosis != ""

    def test_icd10_codes_valid(self):
        agent = DiagnosticAgent()
        ctx = _make_context()
        triage = TriageAgent().assess(ctx)
        result = agent.diagnose(ctx, triage)
        for dx in result.differentials:
            assert dx.icd10_code, f"Missing ICD-10 for {dx.condition}"

    def test_recommends_tests(self):
        agent = DiagnosticAgent()
        ctx = _make_context()
        triage = TriageAgent().assess(ctx)
        result = agent.diagnose(ctx, triage)
        assert len(result.recommended_tests) > 0

    def test_reasoning_chain_populated(self):
        agent = DiagnosticAgent()
        ctx = _make_context()
        triage = TriageAgent().assess(ctx)
        result = agent.diagnose(ctx, triage)
        assert len(result.reasoning_chain) > 0


# ── Treatment Agent ──────────────────────────────────────────

class TestTreatmentAgent:
    def test_generates_treatment_plan(self):
        agent = TreatmentAgent()
        ctx = _make_context()
        triage = TriageAgent().assess(ctx)
        dx = DiagnosticAgent().diagnose(ctx, triage)
        result = agent.plan(ctx, dx)
        assert len(result.treatment_plan) > 0

    def test_recommends_medications(self):
        agent = TreatmentAgent()
        ctx = _make_context()
        triage = TriageAgent().assess(ctx)
        dx = DiagnosticAgent().diagnose(ctx, triage)
        result = agent.plan(ctx, dx)
        assert len(result.medications) > 0

    def test_allergy_check(self):
        agent = TreatmentAgent()
        ctx = _make_context(allergies=["Aspirin"])
        triage = TriageAgent().assess(ctx)
        dx = DiagnosticAgent().diagnose(ctx, triage)
        result = agent.plan(ctx, dx)
        for med in result.medications:
            if "aspirin" in med.name.lower() and "CONTRAINDICATED" not in med.name:
                pytest.fail("Aspirin prescribed despite allergy")

    def test_geriatric_precaution(self):
        agent = TreatmentAgent()
        ctx = _make_context(age=80)
        triage = TriageAgent().assess(ctx)
        dx = DiagnosticAgent().diagnose(ctx, triage)
        result = agent.plan(ctx, dx)
        geriatric = any("geriatric" in p.lower() for p in result.precautions)
        assert geriatric, "Missing geriatric precaution for 80yo patient"

    def test_evidence_grade_present(self):
        agent = TreatmentAgent()
        ctx = _make_context()
        triage = TriageAgent().assess(ctx)
        dx = DiagnosticAgent().diagnose(ctx, triage)
        result = agent.plan(ctx, dx)
        assert result.evidence_grade in ("A", "B", "C", "D")


# ── Orchestrator ─────────────────────────────────────────────

class TestOrchestrator:
    def test_full_pipeline(self):
        orch = ClinicalOrchestrator()
        ctx = _make_context()
        result = orch.reason(ctx)
        assert result.patient_id == "test-001"
        assert result.triage is not None
        assert result.diagnosis is not None
        assert result.treatment is not None
        assert result.pipeline_latency_ms > 0

    def test_audit_trail_complete(self):
        orch = ClinicalOrchestrator()
        ctx = _make_context()
        result = orch.reason(ctx)
        assert len(result.reasoning_audit) == 3
        stages = [a["stage"] for a in result.reasoning_audit]
        assert stages == ["triage", "diagnosis", "treatment"]

    def test_consensus_score_range(self):
        orch = ClinicalOrchestrator()
        ctx = _make_context()
        result = orch.reason(ctx)
        assert 0.0 <= result.consensus_score <= 1.0

    def test_serialization(self):
        orch = ClinicalOrchestrator()
        ctx = _make_context()
        result = orch.reason(ctx)
        d = result.to_dict()
        assert "patient_id" in d
        assert "triage" in d
        assert "diagnosis" in d
        assert "treatment" in d
