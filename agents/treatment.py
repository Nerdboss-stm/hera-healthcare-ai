"""Treatment Agent — Generates evidence-based treatment plans.

Takes the diagnostic output + patient context and produces structured
treatment recommendations including medications, monitoring, disposition,
and follow-up instructions. Includes contraindication checking.
"""

from __future__ import annotations
import logging
from typing import Optional

from agents.protocols import (
    PatientContext, DiagnosticResult, TreatmentResult, Medication,
)

logger = logging.getLogger(__name__)

# Evidence-based treatment protocols keyed by ICD-10
TREATMENT_PROTOCOLS = {
    "I21.9": {  # Acute Coronary Syndrome
        "plan": [
            "Activate cardiac catheterization lab if STEMI",
            "Continuous telemetry monitoring",
            "Serial troponins q3-6h",
            "Cardiology consult",
        ],
        "medications": [
            Medication("Aspirin", "325mg", "PO", "once (loading)", "Antiplatelet", "GI bleed, allergy"),
            Medication("Heparin", "60 units/kg bolus then 12 units/kg/hr", "IV", "continuous", "Anticoagulation", "Active bleeding, HIT"),
            Medication("Nitroglycerin", "0.4mg", "SL", "q5min x3", "Vasodilator / pain relief", "Hypotension, PDE5 inhibitor use"),
            Medication("Atorvastatin", "80mg", "PO", "daily", "High-intensity statin", "Liver disease"),
        ],
        "monitoring": ["Continuous ECG", "BP q15min", "Serial troponins", "I&O"],
        "disposition": "admit",
        "follow_up": "Cardiology follow-up in 1 week post-discharge",
        "evidence_grade": "A",
    },
    "I26.99": {  # Pulmonary Embolism
        "plan": [
            "Anticoagulation therapy",
            "Assess for hemodynamic instability",
            "Consider thrombolysis if massive PE",
            "Hematology consult for unprovoked PE",
        ],
        "medications": [
            Medication("Enoxaparin", "1mg/kg", "SubQ", "q12h", "Anticoagulation", "CrCl <30, active bleeding"),
            Medication("Rivaroxaban", "15mg", "PO", "BID x21 days then 20mg daily", "DOAC transition", "CrCl <15, liver disease"),
        ],
        "monitoring": ["Continuous SpO2", "Repeat CT-PA if worsening", "Renal function"],
        "disposition": "admit",
        "follow_up": "Hematology follow-up in 2 weeks",
        "evidence_grade": "A",
    },
    "J18.9": {  # Pneumonia
        "plan": [
            "Empiric antibiotic therapy per IDSA/ATS guidelines",
            "Supplemental O2 to maintain SpO2 > 94%",
            "Assess CURB-65 score for disposition",
            "Repeat CXR if not improving in 48-72h",
        ],
        "medications": [
            Medication("Ceftriaxone", "1g", "IV", "q24h", "Beta-lactam coverage", "Cephalosporin allergy"),
            Medication("Azithromycin", "500mg", "IV/PO", "daily x5 days", "Atypical coverage", "QT prolongation"),
            Medication("Acetaminophen", "650mg", "PO", "q6h PRN", "Antipyretic", "Liver disease"),
        ],
        "monitoring": ["SpO2 q4h", "Temperature q4h", "Repeat CXR day 2-3"],
        "disposition": "admit",
        "follow_up": "PCP follow-up in 1 week, repeat CXR 6 weeks",
        "evidence_grade": "A",
    },
    "J44.1": {  # COPD Exacerbation
        "plan": [
            "Bronchodilator therapy",
            "Systemic corticosteroids",
            "Antibiotics if purulent sputum",
            "Non-invasive ventilation if respiratory distress",
        ],
        "medications": [
            Medication("Albuterol", "2.5mg", "Nebulizer", "q20min x3 then q4h", "Bronchodilator", "Severe tachycardia"),
            Medication("Ipratropium", "0.5mg", "Nebulizer", "q4h", "Anticholinergic", "Glaucoma, urinary retention"),
            Medication("Prednisone", "40mg", "PO", "daily x5 days", "Anti-inflammatory", "Uncontrolled diabetes"),
            Medication("Azithromycin", "500mg day 1, 250mg days 2-5", "PO", "daily", "Antibiotics", "QT prolongation"),
        ],
        "monitoring": ["SpO2 continuous", "ABG if worsening", "Peak flow q8h"],
        "disposition": "admit",
        "follow_up": "Pulmonology follow-up in 2 weeks",
        "evidence_grade": "A",
    },
    "I50.9": {  # Heart Failure
        "plan": [
            "IV diuresis",
            "Fluid restriction (<1.5L/day)",
            "Daily weights",
            "Cardiology consult",
        ],
        "medications": [
            Medication("Furosemide", "40mg", "IV", "q12h", "Diuresis", "Renal failure, hypokalemia"),
            Medication("Lisinopril", "5mg", "PO", "daily", "ACEi — afterload reduction", "Hyperkalemia, angioedema"),
            Medication("Metoprolol", "25mg", "PO", "BID", "Rate control / remodeling", "Decompensated HF, bradycardia"),
        ],
        "monitoring": ["Daily weights", "I&O strict", "BMP daily", "BNP trend"],
        "disposition": "admit",
        "follow_up": "Cardiology in 1 week, HF clinic enrollment",
        "evidence_grade": "A",
    },
    "A41.9": {  # Sepsis
        "plan": [
            "SEP-1 bundle: lactate, blood cultures, broad-spectrum abx within 1h",
            "30mL/kg crystalloid bolus if hypotensive or lactate >= 4",
            "Vasopressors if refractory hypotension",
            "ICU consult if septic shock",
        ],
        "medications": [
            Medication("Vancomycin", "25-30mg/kg", "IV", "loading dose", "MRSA coverage", "Red man syndrome (infuse slowly)"),
            Medication("Piperacillin-Tazobactam", "4.5g", "IV", "q6h", "Broad gram-negative", "Penicillin allergy"),
            Medication("Normal Saline", "30mL/kg", "IV", "bolus", "Volume resuscitation", "Fluid overload"),
            Medication("Norepinephrine", "0.1-0.5 mcg/kg/min", "IV", "continuous", "Vasopressor", "First-line per SSC guidelines"),
        ],
        "monitoring": ["Lactate q4h until normalizing", "MAP > 65 mmHg", "UOP > 0.5 mL/kg/hr", "CVP if central line"],
        "disposition": "admit",
        "follow_up": "ID consult, narrow antibiotics with culture data",
        "evidence_grade": "A",
    },
    "I63.9": {  # Stroke
        "plan": [
            "Determine onset time — tPA window (< 4.5h)",
            "NIHSS scoring",
            "Neurology / stroke team activation",
            "BP management (permissive HTN if not tPA candidate)",
        ],
        "medications": [
            Medication("Alteplase (tPA)", "0.9mg/kg (max 90mg)", "IV", "10% bolus, 90% over 1h", "Thrombolysis", "Hemorrhage, recent surgery, INR>1.7"),
            Medication("Aspirin", "325mg", "PO", "once (24h after tPA)", "Antiplatelet", "Active bleeding"),
        ],
        "monitoring": ["Neuro checks q15min x2h", "BP q15min during tPA", "CT 24h post-tPA"],
        "disposition": "admit",
        "follow_up": "Neurology follow-up in 1 week, stroke prevention workup",
        "evidence_grade": "A",
    },
    "K35.80": {  # Appendicitis
        "plan": [
            "Surgical consult for appendectomy",
            "NPO",
            "IV antibiotics pre-op",
            "Pain management",
        ],
        "medications": [
            Medication("Cefoxitin", "2g", "IV", "pre-op", "Prophylactic antibiotics", "Cephalosporin allergy"),
            Medication("Morphine", "0.1mg/kg", "IV", "q4h PRN", "Analgesia", "Respiratory depression"),
            Medication("Ondansetron", "4mg", "IV", "q6h PRN", "Antiemetic", "QT prolongation"),
        ],
        "monitoring": ["Serial abdominal exams", "Temperature q4h", "WBC trend"],
        "disposition": "admit",
        "follow_up": "Surgery follow-up in 2 weeks post-op",
        "evidence_grade": "A",
    },
}

# Fallback for diagnoses not in protocol DB
DEFAULT_TREATMENT = {
    "plan": [
        "Supportive care",
        "Monitor vitals q4h",
        "Symptom management",
        "Specialist consult as indicated",
    ],
    "medications": [
        Medication("Acetaminophen", "650mg", "PO", "q6h PRN", "Analgesia/Antipyretic", "Liver disease"),
    ],
    "monitoring": ["Vital signs q4h", "Clinical reassessment in 4-6h"],
    "disposition": "observe",
    "follow_up": "PCP follow-up in 3-5 days",
    "evidence_grade": "C",
}

# Known drug interactions (simplified)
DRUG_INTERACTIONS = {
    ("Nitroglycerin", "PDE5 inhibitor"): "CONTRAINDICATED: Severe hypotension risk",
    ("Heparin", "Enoxaparin"): "Avoid dual anticoagulation",
    ("Metoprolol", "Verapamil"): "Risk of severe bradycardia/AV block",
    ("Lisinopril", "Potassium"): "Monitor potassium — hyperkalemia risk",
    ("Warfarin", "Aspirin"): "Increased bleeding risk — monitor closely",
}


class TreatmentAgent:
    """Generates evidence-based treatment plans from diagnostic results.

    Matches primary diagnosis to treatment protocols, checks for
    contraindications against patient context, and produces a
    structured plan with medications, monitoring, and disposition.
    """

    def plan(
        self, ctx: PatientContext, dx: DiagnosticResult
    ) -> TreatmentResult:
        logger.info("Treatment planning for patient %s", ctx.patient_id)

        icd_code = dx.differentials[0].icd10_code if dx.differentials else None
        protocol = TREATMENT_PROTOCOLS.get(icd_code, DEFAULT_TREATMENT)

        # Filter medications against patient allergies
        safe_meds = self._check_contraindications(
            protocol["medications"], ctx
        )

        # Check drug interactions with current medications
        interaction_warnings = self._check_interactions(safe_meds, ctx)
        precautions = interaction_warnings.copy()

        # Age-based precautions
        if ctx.age > 75:
            precautions.append(
                "Geriatric patient — consider renal dosing, fall risk, delirium screening"
            )
        if ctx.age < 18:
            precautions.append(
                "Pediatric patient — verify weight-based dosing"
            )

        # Adjust disposition based on acuity
        disposition = protocol["disposition"]
        if dx.confidence < 0.3 and disposition == "admit":
            disposition = "observe"

        return TreatmentResult(
            treatment_plan=protocol["plan"],
            medications=safe_meds,
            monitoring_plan=protocol["monitoring"],
            disposition=disposition,
            follow_up=protocol["follow_up"],
            precautions=precautions,
            evidence_grade=protocol["evidence_grade"],
        )

    def _check_contraindications(
        self, medications: list[Medication], ctx: PatientContext
    ) -> list[Medication]:
        safe = []
        for med in medications:
            allergic = any(
                allergy.lower() in med.name.lower()
                for allergy in ctx.allergies
            )
            if allergic:
                logger.warning(
                    "ALLERGY: Skipping %s for patient %s (allergy: %s)",
                    med.name, ctx.patient_id, ctx.allergies,
                )
                safe.append(Medication(
                    name=f"[CONTRAINDICATED: {med.name}]",
                    dose="N/A",
                    route="N/A",
                    frequency="N/A",
                    rationale=f"Patient allergic — use alternative. Original: {med.rationale}",
                    contraindication_check=f"ALLERGY DETECTED: {', '.join(ctx.allergies)}",
                ))
            else:
                safe.append(med)
        return safe

    def _check_interactions(
        self, proposed: list[Medication], ctx: PatientContext
    ) -> list[str]:
        warnings = []
        current_lower = [m.lower() for m in ctx.current_medications]
        for med in proposed:
            for (drug_a, drug_b), warning in DRUG_INTERACTIONS.items():
                if (
                    drug_a.lower() in med.name.lower()
                    and any(drug_b.lower() in cm for cm in current_lower)
                ):
                    warnings.append(f"{med.name}: {warning}")
                elif (
                    drug_b.lower() in med.name.lower()
                    and any(drug_a.lower() in cm for cm in current_lower)
                ):
                    warnings.append(f"{med.name}: {warning}")
        return warnings
