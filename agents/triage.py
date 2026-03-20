"""Triage Agent — Assigns ESI acuity level based on vitals and chief complaint.

Uses rule-based clinical decision logic aligned with the ESI v4 algorithm,
augmented by the trained risk prediction model for vital sign analysis.
"""

from __future__ import annotations
import logging
from typing import Optional

from agents.protocols import PatientContext, TriageResult, ESILevel

logger = logging.getLogger(__name__)

# Clinical thresholds based on ESI v4 / NEWS2
VITAL_THRESHOLDS = {
    "heart_rate": {"critical_low": 40, "low": 51, "high": 110, "critical_high": 130},
    "respiratory_rate": {"critical_low": 8, "low": 9, "high": 21, "critical_high": 25},
    "body_temperature": {"critical_low": 35.0, "low": 36.1, "high": 38.0, "critical_high": 39.0},
    "oxygen_saturation": {"critical_low": 88, "low": 92, "high": 100, "critical_high": 101},
    "systolic_bp": {"critical_low": 80, "low": 90, "high": 160, "critical_high": 180},
    "diastolic_bp": {"critical_low": 40, "low": 60, "high": 100, "critical_high": 110},
}

# Chief complaints that auto-escalate ESI level
HIGH_ACUITY_COMPLAINTS = {
    1: ["cardiac arrest", "respiratory arrest", "unresponsive", "pulseless",
        "apneic", "intubation", "code blue"],
    2: ["chest pain", "stroke", "stemi", "anaphylaxis", "sepsis",
        "altered mental status", "overdose", "gi bleed", "hemorrhage",
        "severe trauma", "status epilepticus", "acute mi"],
}

# Expected resource counts by complaint category
RESOURCE_ESTIMATES = {
    "labs_imaging": ["abdominal pain", "headache", "fever", "back pain",
                     "weakness", "dizziness", "urinary"],
    "single_resource": ["laceration", "sprain", "rash", "sore throat",
                        "ear pain", "refill", "suture removal"],
    "no_resources": ["prescription refill", "medical clearance",
                     "wound check", "suture removal"],
}


class TriageAgent:
    """Classifies patient acuity using the ESI v4 triage algorithm.

    Analyzes vitals for out-of-range flags, maps chief complaint to acuity,
    and produces a structured TriageResult with rationale.
    """

    def __init__(self, risk_predictor=None):
        self._risk_predictor = risk_predictor

    def assess(self, ctx: PatientContext) -> TriageResult:
        logger.info("Triage assessment for patient %s", ctx.patient_id)

        vital_flags = self._check_vitals(ctx.vitals)
        complaint_esi = self._complaint_acuity(ctx.chief_complaint)
        vital_esi = self._vital_acuity(vital_flags)
        resource_esi = self._resource_estimate(ctx.chief_complaint)

        # ESI = most acute (lowest number) of all signals
        esi = min(complaint_esi, vital_esi, resource_esi)
        risk_score = self._compute_risk_score(ctx, vital_flags, esi)

        immediate_actions = self._immediate_actions(esi, vital_flags, ctx)
        rationale = self._build_rationale(
            esi, complaint_esi, vital_esi, resource_esi, vital_flags, ctx
        )

        return TriageResult(
            esi_level=ESILevel(esi),
            acuity_rationale=rationale,
            vital_flags=vital_flags,
            immediate_actions=immediate_actions,
            risk_score=risk_score,
        )

    def _check_vitals(self, vitals: dict) -> list[str]:
        flags = []
        for param, thresholds in VITAL_THRESHOLDS.items():
            value = vitals.get(param)
            if value is None:
                continue
            if value <= thresholds["critical_low"]:
                flags.append(f"CRITICAL: {param}={value} (critically low)")
            elif value <= thresholds["low"]:
                flags.append(f"WARNING: {param}={value} (below normal)")
            elif param == "oxygen_saturation" and value < thresholds["low"]:
                flags.append(f"CRITICAL: {param}={value}% (hypoxia)")
            elif value >= thresholds["critical_high"]:
                flags.append(f"CRITICAL: {param}={value} (critically high)")
            elif value >= thresholds["high"]:
                flags.append(f"WARNING: {param}={value} (above normal)")
        return flags

    def _complaint_acuity(self, complaint: str) -> int:
        complaint_lower = complaint.lower()
        for esi_level, keywords in HIGH_ACUITY_COMPLAINTS.items():
            if any(kw in complaint_lower for kw in keywords):
                return esi_level
        return 5  # default — will be overridden by resource estimate

    def _vital_acuity(self, flags: list[str]) -> int:
        critical_count = sum(1 for f in flags if f.startswith("CRITICAL"))
        warning_count = sum(1 for f in flags if f.startswith("WARNING"))
        if critical_count >= 2:
            return 1
        if critical_count >= 1:
            return 2
        if warning_count >= 2:
            return 3
        if warning_count >= 1:
            return 4
        return 5

    def _resource_estimate(self, complaint: str) -> int:
        complaint_lower = complaint.lower()
        for kw in RESOURCE_ESTIMATES["no_resources"]:
            if kw in complaint_lower:
                return 5
        for kw in RESOURCE_ESTIMATES["single_resource"]:
            if kw in complaint_lower:
                return 4
        for kw in RESOURCE_ESTIMATES["labs_imaging"]:
            if kw in complaint_lower:
                return 3
        return 3  # default: assume labs/imaging needed

    def _compute_risk_score(
        self, ctx: PatientContext, flags: list[str], esi: int
    ) -> float:
        # Use ML risk predictor if available
        if self._risk_predictor:
            try:
                result = self._risk_predictor(
                    heart_rate=ctx.vitals.get("heart_rate", 80),
                    respiratory_rate=ctx.vitals.get("respiratory_rate", 16),
                    body_temperature=ctx.vitals.get("body_temperature", 37.0),
                    oxygen_saturation=ctx.vitals.get("oxygen_saturation", 98),
                    systolic_bp=ctx.vitals.get("systolic_bp", 120),
                    diastolic_bp=ctx.vitals.get("diastolic_bp", 80),
                    age=ctx.age,
                )
                return result.get("risk_score", 0.5)
            except Exception:
                logger.warning("ML risk predictor failed, using heuristic")

        # Heuristic fallback
        critical_count = sum(1 for f in flags if f.startswith("CRITICAL"))
        warning_count = sum(1 for f in flags if f.startswith("WARNING"))
        base = (5 - esi) / 4.0
        vital_penalty = min(critical_count * 0.2 + warning_count * 0.1, 0.4)
        age_factor = 0.1 if ctx.age > 65 else 0.0
        return min(base + vital_penalty + age_factor, 1.0)

    def _immediate_actions(
        self, esi: int, flags: list[str], ctx: PatientContext
    ) -> list[str]:
        actions = []
        if esi == 1:
            actions.extend([
                "Activate resuscitation team",
                "Establish IV access immediately",
                "Continuous cardiac monitoring",
                "Prepare airway management equipment",
            ])
        elif esi == 2:
            actions.extend([
                "Place in monitored bed",
                "Establish IV access",
                "12-lead ECG within 10 minutes",
                "Notify attending physician",
            ])
        elif esi == 3:
            actions.extend([
                "Obtain full vital signs",
                "Order initial labs (CBC, BMP, troponin if indicated)",
            ])

        if any("oxygen_saturation" in f for f in flags):
            actions.append("Apply supplemental O2, titrate to SpO2 > 94%")
        if any("body_temperature" in f and "high" in f for f in flags):
            actions.append("Administer antipyretic, obtain blood cultures")
        if any("systolic_bp" in f and "low" in f.lower() for f in flags):
            actions.append("Bolus NS 500mL IV, reassess after")

        return actions

    def _build_rationale(
        self, esi, complaint_esi, vital_esi, resource_esi, flags, ctx
    ) -> str:
        parts = [f"Final ESI Level: {esi}."]
        if complaint_esi <= 2:
            parts.append(
                f"Chief complaint '{ctx.chief_complaint}' maps to ESI-{complaint_esi}."
            )
        if vital_esi < 5:
            parts.append(
                f"Vital sign analysis yields ESI-{vital_esi} "
                f"({len(flags)} flag(s) detected)."
            )
        if resource_esi >= 4:
            parts.append(
                f"Resource estimate: ESI-{resource_esi} "
                f"(low resource utilization expected)."
            )
        parts.append(
            f"Patient: {ctx.age}y {ctx.gender}, "
            f"presenting with '{ctx.chief_complaint}'."
        )
        return " ".join(parts)
