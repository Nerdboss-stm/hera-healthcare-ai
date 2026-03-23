"""Diagnostic Agent — Generates ranked differential diagnoses.

Combines clinical note analysis, vital sign patterns, triage context,
and a medical knowledge base to produce probable diagnoses with ICD-10
codes, supporting evidence, and recommended confirmatory tests.
"""

from __future__ import annotations
import logging

from agents.protocols import (
    PatientContext,
    TriageResult,
    DiagnosticResult,
    Diagnosis,
)

logger = logging.getLogger(__name__)

# Symptom-to-diagnosis mapping (expandable via RAG in future)
CLINICAL_KNOWLEDGE = {
    "chest pain": [
        Diagnosis(
            condition="Acute Coronary Syndrome",
            icd10_code="I21.9",
            probability=0.0,
            supporting_evidence=["chest pain", "diaphoresis", "radiation to arm/jaw"],
            ruling_out=["normal troponin", "normal ECG", "pleuritic quality"],
        ),
        Diagnosis(
            condition="Pulmonary Embolism",
            icd10_code="I26.99",
            probability=0.0,
            supporting_evidence=[
                "pleuritic chest pain",
                "dyspnea",
                "tachycardia",
                "hypoxia",
            ],
            ruling_out=["negative D-dimer", "normal CT-PA"],
        ),
        Diagnosis(
            condition="Pneumothorax",
            icd10_code="J93.9",
            probability=0.0,
            supporting_evidence=[
                "sudden onset",
                "pleuritic",
                "decreased breath sounds",
            ],
            ruling_out=["normal chest X-ray"],
        ),
        Diagnosis(
            condition="Costochondritis",
            icd10_code="M94.0",
            probability=0.0,
            supporting_evidence=["reproducible on palpation", "sharp", "localized"],
            ruling_out=["cardiac biomarkers positive"],
        ),
    ],
    "dyspnea": [
        Diagnosis(
            condition="Acute Exacerbation of COPD",
            icd10_code="J44.1",
            probability=0.0,
            supporting_evidence=["wheezing", "productive cough", "smoking history"],
            ruling_out=["no prior COPD diagnosis"],
        ),
        Diagnosis(
            condition="Congestive Heart Failure",
            icd10_code="I50.9",
            probability=0.0,
            supporting_evidence=["orthopnea", "edema", "JVD", "crackles"],
            ruling_out=["normal BNP", "normal echo"],
        ),
        Diagnosis(
            condition="Pneumonia",
            icd10_code="J18.9",
            probability=0.0,
            supporting_evidence=[
                "fever",
                "productive cough",
                "crackles",
                "consolidation",
            ],
            ruling_out=["clear chest X-ray", "no fever"],
        ),
    ],
    "headache": [
        Diagnosis(
            condition="Migraine",
            icd10_code="G43.909",
            probability=0.0,
            supporting_evidence=["unilateral", "pulsating", "photophobia", "nausea"],
            ruling_out=["sudden thunderclap onset", "focal neuro deficits"],
        ),
        Diagnosis(
            condition="Subarachnoid Hemorrhage",
            icd10_code="I60.9",
            probability=0.0,
            supporting_evidence=[
                "thunderclap onset",
                "worst headache of life",
                "neck stiffness",
            ],
            ruling_out=["gradual onset", "normal CT head"],
        ),
        Diagnosis(
            condition="Tension-Type Headache",
            icd10_code="G44.209",
            probability=0.0,
            supporting_evidence=["bilateral", "pressing quality", "mild-moderate"],
            ruling_out=["severe intensity", "focal signs"],
        ),
    ],
    "abdominal pain": [
        Diagnosis(
            condition="Acute Appendicitis",
            icd10_code="K35.80",
            probability=0.0,
            supporting_evidence=["RLQ pain", "rebound tenderness", "fever", "anorexia"],
            ruling_out=["normal CT abdomen", "pain resolves"],
        ),
        Diagnosis(
            condition="Acute Cholecystitis",
            icd10_code="K81.0",
            probability=0.0,
            supporting_evidence=[
                "RUQ pain",
                "Murphy's sign",
                "post-prandial",
                "nausea",
            ],
            ruling_out=["normal ultrasound", "no gallstones"],
        ),
        Diagnosis(
            condition="Small Bowel Obstruction",
            icd10_code="K56.60",
            probability=0.0,
            supporting_evidence=[
                "distension",
                "vomiting",
                "obstipation",
                "prior surgery",
            ],
            ruling_out=["normal abdominal X-ray", "passing flatus"],
        ),
    ],
    "fever": [
        Diagnosis(
            condition="Sepsis",
            icd10_code="A41.9",
            probability=0.0,
            supporting_evidence=["fever", "tachycardia", "hypotension", "elevated WBC"],
            ruling_out=["hemodynamically stable", "localizing source found"],
        ),
        Diagnosis(
            condition="Urinary Tract Infection",
            icd10_code="N39.0",
            probability=0.0,
            supporting_evidence=["dysuria", "frequency", "suprapubic pain"],
            ruling_out=["clean UA"],
        ),
        Diagnosis(
            condition="Community-Acquired Pneumonia",
            icd10_code="J18.9",
            probability=0.0,
            supporting_evidence=["cough", "fever", "crackles", "infiltrate on CXR"],
            ruling_out=["clear chest X-ray"],
        ),
    ],
    "altered mental status": [
        Diagnosis(
            condition="Hypoglycemia",
            icd10_code="E16.2",
            probability=0.0,
            supporting_evidence=[
                "diabetes history",
                "diaphoresis",
                "tremor",
                "low glucose",
            ],
            ruling_out=["normal glucose"],
        ),
        Diagnosis(
            condition="Stroke (CVA)",
            icd10_code="I63.9",
            probability=0.0,
            supporting_evidence=[
                "focal deficits",
                "sudden onset",
                "facial droop",
                "arm drift",
            ],
            ruling_out=["normal CT/MRI", "no focal findings"],
        ),
        Diagnosis(
            condition="Drug Intoxication",
            icd10_code="T50.905A",
            probability=0.0,
            supporting_evidence=["substance use history", "pupil changes", "toxidrome"],
            ruling_out=["negative tox screen"],
        ),
    ],
}

# Symptom aliases — maps common phrases to CLINICAL_KNOWLEDGE keys
SYMPTOM_ALIASES: dict[str, str] = {
    "shortness of breath": "dyspnea",
    "sob": "dyspnea",
    "difficulty breathing": "dyspnea",
    "breathing difficulty": "dyspnea",
    "breathless": "dyspnea",
    "can't breathe": "dyspnea",
    "cough": "dyspnea",
    "wheezing": "dyspnea",
    "chest tightness": "chest pain",
    "heart attack": "chest pain",
    "substernal": "chest pain",
    "angina": "chest pain",
    "palpitations": "chest pain",
    "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "nausea": "abdominal pain",
    "vomiting": "abdominal pain",
    "diarrhea": "abdominal pain",
    "constipation": "abdominal pain",
    "migraine": "headache",
    "head pain": "headache",
    "dizziness": "headache",
    "dizzy": "headache",
    "vertigo": "headache",
    "high temperature": "fever",
    "chills": "fever",
    "infection": "fever",
    "septic": "fever",
    "confused": "altered mental status",
    "confusion": "altered mental status",
    "unresponsive": "altered mental status",
    "drowsy": "altered mental status",
    "syncope": "altered mental status",
    "fainted": "altered mental status",
    "passed out": "altered mental status",
    "seizure": "altered mental status",
    "unconscious": "altered mental status",
    "weak": "altered mental status",
    "weakness": "altered mental status",
    "fall": "altered mental status",
    "back pain": "abdominal pain",
    "flank pain": "abdominal pain",
    "urinary": "fever",
    "dysuria": "fever",
    "rash": "fever",
    "swelling": "abdominal pain",
    "bleeding": "abdominal pain",
    "laceration": "chest pain",
    "trauma": "chest pain",
}

# Confirmatory tests by diagnosis category
RECOMMENDED_TESTS = {
    "I21.9": ["Troponin (serial q3h)", "12-lead ECG", "CXR", "BMP"],
    "I26.99": ["D-dimer", "CT Pulmonary Angiography", "ABG"],
    "J93.9": ["CXR (upright)", "CT chest if uncertain"],
    "I50.9": ["BNP/NT-proBNP", "CXR", "Echocardiogram", "BMP"],
    "J18.9": ["CXR", "CBC", "Blood cultures", "Procalcitonin"],
    "J44.1": ["ABG", "CXR", "CBC", "BMP"],
    "A41.9": ["Blood cultures x2", "Lactate", "CBC", "BMP", "Procalcitonin", "UA"],
    "I60.9": ["CT Head (non-contrast)", "LP if CT negative", "CTA Head/Neck"],
    "K35.80": ["CT Abdomen/Pelvis with contrast", "CBC", "CRP"],
    "K81.0": ["RUQ Ultrasound", "CBC", "LFTs", "Lipase"],
    "I63.9": ["CT Head", "CT Angiography", "MRI DWI", "CBC", "BMP", "Glucose"],
    "N39.0": ["Urinalysis", "Urine culture"],
    "E16.2": ["Point-of-care glucose", "BMP"],
}


class DiagnosticAgent:
    """Generates ranked differential diagnoses from patient context + triage.

    Matches symptoms from the clinical note against a medical knowledge base,
    scores probabilities using vital-sign signals and symptom overlap,
    and recommends confirmatory tests for the top differentials.
    """

    def __init__(self, summarizer=None):
        self._summarizer = summarizer

    def diagnose(self, ctx: PatientContext, triage: TriageResult) -> DiagnosticResult:
        logger.info("Diagnostic assessment for patient %s", ctx.patient_id)

        note_lower = ctx.clinical_note.lower()
        complaint_lower = ctx.chief_complaint.lower()
        combined_text = f"{complaint_lower} {note_lower}"

        # Resolve symptom aliases to canonical category names
        resolved_categories: set[str] = set()
        for alias, canonical in SYMPTOM_ALIASES.items():
            if alias in combined_text:
                resolved_categories.add(canonical)

        # Collect candidate diagnoses from all matching symptom categories
        candidates: list[Diagnosis] = []
        matched_categories = set()
        for category, diagnoses in CLINICAL_KNOWLEDGE.items():
            if (
                category in complaint_lower
                or category in note_lower
                or category in resolved_categories
            ):
                matched_categories.add(category)
                for dx in diagnoses:
                    candidates.append(
                        Diagnosis(
                            condition=dx.condition,
                            icd10_code=dx.icd10_code,
                            probability=0.0,
                            supporting_evidence=list(dx.supporting_evidence),
                            ruling_out=list(dx.ruling_out),
                        )
                    )

        if not candidates:
            # Fallback: use chief complaint keywords to find nearest match
            for category in CLINICAL_KNOWLEDGE:
                words = category.split()
                if any(w in complaint_lower for w in words):
                    for dx in CLINICAL_KNOWLEDGE[category]:
                        candidates.append(
                            Diagnosis(
                                condition=dx.condition,
                                icd10_code=dx.icd10_code,
                                probability=0.0,
                                supporting_evidence=list(dx.supporting_evidence),
                                ruling_out=list(dx.ruling_out),
                            )
                        )
                    break

        # Score each candidate
        reasoning_chain = []
        for dx in candidates:
            evidence_found = []
            ruling_out_found = []
            for ev in dx.supporting_evidence:
                if ev.lower() in note_lower:
                    evidence_found.append(ev)
            for ro in dx.ruling_out:
                if ro.lower() in note_lower:
                    ruling_out_found.append(ro)

            # Probability = evidence overlap minus rule-out overlap
            evidence_score = len(evidence_found) / max(len(dx.supporting_evidence), 1)
            ruling_penalty = len(ruling_out_found) * 0.3
            acuity_boost = 0.1 if triage.esi_level.value <= 2 else 0.0
            age_factor = 0.05 if ctx.age > 60 else 0.0

            dx.probability = round(
                min(
                    max(
                        evidence_score - ruling_penalty + acuity_boost + age_factor,
                        0.05,
                    ),
                    0.95,
                ),
                3,
            )
            dx.supporting_evidence = evidence_found or dx.supporting_evidence[:2]

            reasoning_chain.append(
                f"{dx.condition} ({dx.icd10_code}): "
                f"evidence={len(evidence_found)}/{len(dx.supporting_evidence)}, "
                f"rule-outs={len(ruling_out_found)}, "
                f"P={dx.probability}"
            )

        # Sort by probability descending
        candidates.sort(key=lambda d: d.probability, reverse=True)

        # Collect recommended tests from top 3
        tests = []
        seen_tests = set()
        for dx in candidates[:3]:
            for test in RECOMMENDED_TESTS.get(dx.icd10_code, []):
                if test not in seen_tests:
                    tests.append(test)
                    seen_tests.add(test)

        primary = candidates[0] if candidates else None

        return DiagnosticResult(
            differentials=candidates[:5],
            primary_diagnosis=primary.condition if primary else "Undifferentiated",
            confidence=primary.probability if primary else 0.1,
            reasoning_chain=reasoning_chain,
            recommended_tests=tests[:10],
        )
