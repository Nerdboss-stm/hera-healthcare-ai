"""Clinical Evaluator — Multi-dimensional AI output evaluation.

Implements factual consistency checking, hallucination detection,
medical term validation, and entity-level accuracy scoring.
Goes far beyond ROUGE to catch clinically dangerous errors.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FactualConsistencyResult:
    """Result of factual consistency check between source and generated text."""
    score: float  # 0.0 (inconsistent) to 1.0 (fully consistent)
    supported_claims: list[str]
    unsupported_claims: list[str]
    contradicted_claims: list[str]


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    hallucination_score: float  # 0.0 (no hallucination) to 1.0 (fully hallucinated)
    hallucinated_entities: list[str]
    hallucinated_claims: list[str]
    entity_precision: float
    entity_recall: float


@dataclass
class MedicalAccuracyResult:
    """Result of medical terminology and logic validation."""
    accuracy_score: float
    valid_terms: list[str]
    invalid_terms: list[str]
    dosage_errors: list[str]
    logic_errors: list[str]


@dataclass
class ClinicalSafetyResult:
    """Result of clinical safety check."""
    safe: bool
    severity: str  # "none", "low", "medium", "high", "critical"
    issues: list[str]


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a clinical AI output."""
    factual_consistency: FactualConsistencyResult
    hallucination: HallucinationResult
    medical_accuracy: MedicalAccuracyResult
    clinical_safety: ClinicalSafetyResult
    overall_score: float
    pass_threshold: bool
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "pass_threshold": self.pass_threshold,
            "factual_consistency": {
                "score": self.factual_consistency.score,
                "supported": len(self.factual_consistency.supported_claims),
                "unsupported": len(self.factual_consistency.unsupported_claims),
                "contradicted": len(self.factual_consistency.contradicted_claims),
            },
            "hallucination": {
                "score": self.hallucination.hallucination_score,
                "entities_hallucinated": len(self.hallucination.hallucinated_entities),
                "entity_precision": self.hallucination.entity_precision,
                "entity_recall": self.hallucination.entity_recall,
            },
            "medical_accuracy": {
                "score": self.medical_accuracy.accuracy_score,
                "valid_terms": len(self.medical_accuracy.valid_terms),
                "invalid_terms": len(self.medical_accuracy.invalid_terms),
                "dosage_errors": len(self.medical_accuracy.dosage_errors),
            },
            "clinical_safety": {
                "safe": self.clinical_safety.safe,
                "severity": self.clinical_safety.severity,
                "issues": self.clinical_safety.issues,
            },
        }


# Known valid medical terms (expandable)
VALID_MEDICAL_TERMS = {
    "hypertension", "hypotension", "tachycardia", "bradycardia",
    "tachypnea", "dyspnea", "hypoxia", "hypoxemia", "cyanosis",
    "edema", "effusion", "consolidation", "infiltrate", "atelectasis",
    "pneumothorax", "hemothorax", "pneumonia", "bronchitis",
    "myocardial infarction", "angina", "arrhythmia", "fibrillation",
    "embolism", "thrombosis", "hemorrhage", "aneurysm",
    "diabetes", "ketoacidosis", "hyperglycemia", "hypoglycemia",
    "sepsis", "bacteremia", "cellulitis", "abscess",
    "fracture", "dislocation", "contusion", "laceration",
    "stroke", "ischemia", "infarction", "hemorrhagic",
    "renal failure", "hepatic failure", "cirrhosis",
    "appendicitis", "cholecystitis", "pancreatitis", "peritonitis",
    "meningitis", "encephalitis", "seizure", "epilepsy",
    "anaphylaxis", "urticaria", "angioedema",
}

# Dangerous dosage ranges that should trigger safety alerts
DANGEROUS_DOSAGES = {
    "morphine": {"max_single_iv": 15, "unit": "mg"},
    "heparin": {"max_bolus": 10000, "unit": "units"},
    "insulin": {"max_single": 50, "unit": "units"},
    "potassium": {"max_iv_rate": 20, "unit": "mEq/hr"},
    "epinephrine": {"max_iv": 1, "unit": "mg"},
}

# Contradiction patterns
CONTRADICTION_PAIRS = [
    ("improved", "worsened"),
    ("stable", "unstable"),
    ("normal", "abnormal"),
    ("negative", "positive"),
    ("benign", "malignant"),
    ("acute", "chronic"),
    ("increased", "decreased"),
]


class ClinicalEvaluator:
    """Multi-dimensional evaluator for clinical AI outputs.

    Checks factual consistency, hallucinations, medical accuracy,
    and clinical safety. Produces a comprehensive evaluation report
    with an overall quality score.
    """

    def __init__(self, pass_threshold: float = 0.7):
        self._threshold = pass_threshold

    def evaluate(
        self,
        source_note: str,
        generated_output: str,
        context: dict | None = None,
    ) -> EvaluationReport:
        """Run full evaluation pipeline on a generated clinical output."""
        factual = self._check_factual_consistency(source_note, generated_output)
        hallucination = self._detect_hallucinations(source_note, generated_output)
        accuracy = self._validate_medical_accuracy(generated_output)
        safety = self._check_clinical_safety(generated_output, context)

        # Weighted overall score
        overall = (
            0.30 * factual.score
            + 0.25 * (1.0 - hallucination.hallucination_score)
            + 0.25 * accuracy.accuracy_score
            + 0.20 * (1.0 if safety.safe else 0.0)
        )

        return EvaluationReport(
            factual_consistency=factual,
            hallucination=hallucination,
            medical_accuracy=accuracy,
            clinical_safety=safety,
            overall_score=round(overall, 3),
            pass_threshold=overall >= self._threshold,
        )

    def _check_factual_consistency(
        self, source: str, generated: str
    ) -> FactualConsistencyResult:
        """Check if generated text is factually consistent with source.

        Uses entity overlap and claim-level verification.
        """
        source_lower = source.lower()

        # Extract claims (sentences) from generated text
        claims = [s.strip() for s in re.split(r'[.!?]+', generated) if len(s.strip()) > 10]

        supported = []
        unsupported = []
        contradicted = []

        for claim in claims:
            claim_lower = claim.lower()

            # Check for contradiction patterns
            is_contradicted = False
            for pos, neg in CONTRADICTION_PAIRS:
                if pos in claim_lower and neg in source_lower:
                    # Check if they refer to the same entity
                    claim_words = set(claim_lower.split())
                    source_words = set(source_lower.split())
                    overlap = claim_words & source_words
                    if len(overlap) > 3:  # significant overlap suggests same context
                        contradicted.append(claim)
                        is_contradicted = True
                        break

            if is_contradicted:
                continue

            # Check keyword overlap for support
            claim_keywords = set(re.findall(r'\b[a-z]{4,}\b', claim_lower))
            source_keywords = set(re.findall(r'\b[a-z]{4,}\b', source_lower))
            if not claim_keywords:
                continue

            overlap_ratio = len(claim_keywords & source_keywords) / len(claim_keywords)
            if overlap_ratio >= 0.5:
                supported.append(claim)
            else:
                unsupported.append(claim)

        total = len(supported) + len(unsupported) + len(contradicted)
        score = len(supported) / max(total, 1)

        # Contradictions heavily penalize the score
        if contradicted:
            score = max(score - 0.3 * len(contradicted), 0.0)

        return FactualConsistencyResult(
            score=round(score, 3),
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
        )

    def _detect_hallucinations(
        self, source: str, generated: str
    ) -> HallucinationResult:
        """Detect hallucinated entities and claims in generated text."""
        source_lower = source.lower()

        # Extract medical entities from both texts
        med_pattern = (
            r'\b(?:mg|mcg|mL|units?|tablets?|capsules?|'
            r'[A-Z][a-z]+(?:ine|ole|cin|tin|mab|nib|pril|lol|tan|pin))\b'
        )
        source_entities = set(re.findall(med_pattern, source, re.IGNORECASE))
        generated_entities = set(re.findall(med_pattern, generated, re.IGNORECASE))

        # Also extract numbers with units
        num_pattern = r'\d+\.?\d*\s*(?:mg|mcg|g|mL|%|mmHg|bpm|°[CF])'
        source_nums = set(re.findall(num_pattern, source, re.IGNORECASE))
        generated_nums = set(re.findall(num_pattern, generated, re.IGNORECASE))

        all_source = source_entities | source_nums
        all_generated = generated_entities | generated_nums

        # Hallucinated = in generated but not in source
        hallucinated_entities = list(all_generated - all_source)

        # Entity precision and recall
        true_positives = len(all_generated & all_source)
        precision = true_positives / max(len(all_generated), 1)
        recall = true_positives / max(len(all_source), 1)

        # Check for fabricated claims
        hallucinated_claims = []
        sentences = [s.strip() for s in re.split(r'[.!?]+', generated) if len(s.strip()) > 15]
        for sent in sentences:
            sent_lower = sent.lower()
            # If a sentence has very low overlap with source, it may be hallucinated
            words = set(re.findall(r'\b[a-z]{4,}\b', sent_lower))
            source_words = set(re.findall(r'\b[a-z]{4,}\b', source_lower))
            if words and len(words & source_words) / len(words) < 0.2:
                hallucinated_claims.append(sent)

        hallucination_score = len(hallucinated_entities) / max(len(all_generated), 1)
        if hallucinated_claims:
            hallucination_score = min(
                hallucination_score + 0.1 * len(hallucinated_claims), 1.0
            )

        return HallucinationResult(
            hallucination_score=round(hallucination_score, 3),
            hallucinated_entities=hallucinated_entities[:10],
            hallucinated_claims=hallucinated_claims[:5],
            entity_precision=round(precision, 3),
            entity_recall=round(recall, 3),
        )

    def _validate_medical_accuracy(self, text: str) -> MedicalAccuracyResult:
        """Validate medical terminology and dosage accuracy."""
        text_lower = text.lower()

        valid_terms = []
        invalid_terms = []

        # Check for known medical terms
        for term in VALID_MEDICAL_TERMS:
            if term in text_lower:
                valid_terms.append(term)

        # Check for potential misspellings or nonsensical medical terms
        # Look for words ending in common medical suffixes
        medical_words = re.findall(
            r'\b[a-z]+(?:itis|osis|emia|pathy|ectomy|otomy|plasty|scopy|gram|graph)\b',
            text_lower,
        )
        for word in medical_words:
            if word not in VALID_MEDICAL_TERMS and word not in text_lower:
                invalid_terms.append(word)

        # Check dosages
        dosage_errors = []
        dosage_matches = re.findall(
            r'(\w+)\s+(\d+\.?\d*)\s*(mg|mcg|units?|mEq)',
            text, re.IGNORECASE,
        )
        for drug, dose_str, unit in dosage_matches:
            drug_lower = drug.lower()
            dose = float(dose_str)
            if drug_lower in DANGEROUS_DOSAGES:
                limits = DANGEROUS_DOSAGES[drug_lower]
                max_val = list(limits.values())[0]
                if isinstance(max_val, (int, float)) and dose > max_val * 2:
                    dosage_errors.append(
                        f"Potentially dangerous: {drug} {dose}{unit} "
                        f"(expected max ~{max_val}{limits.get('unit', '')})"
                    )

        # Logic errors
        logic_errors = []
        if "administer" in text_lower and "allergic" in text_lower:
            logic_errors.append("Potential: administering medication to allergic patient")

        total_checks = max(len(valid_terms) + len(invalid_terms), 1)
        accuracy = len(valid_terms) / total_checks
        if dosage_errors:
            accuracy = max(accuracy - 0.1 * len(dosage_errors), 0.0)

        return MedicalAccuracyResult(
            accuracy_score=round(accuracy, 3),
            valid_terms=valid_terms,
            invalid_terms=invalid_terms,
            dosage_errors=dosage_errors,
            logic_errors=logic_errors,
        )

    def _check_clinical_safety(
        self, text: str, context: dict | None = None
    ) -> ClinicalSafetyResult:
        """Check for clinically dangerous content in generated output."""
        issues = []
        text_lower = text.lower()

        # Check for dangerous recommendations
        dangerous_patterns = [
            (r"discontinue\s+all\s+medications", "Recommending discontinuation of all medications"),
            (r"no\s+(?:further|additional)\s+(?:treatment|care)\s+(?:needed|required)",
             "Potentially dismissing need for care"),
            (r"discharge\s+(?:immediately|now).*(?:chest pain|stroke|sepsis)",
             "Recommending discharge for critical condition"),
        ]
        for pattern, description in dangerous_patterns:
            if re.search(pattern, text_lower):
                issues.append(description)

        # Check for missing critical warnings
        if context:
            allergies = context.get("allergies", [])
            for allergy in allergies:
                if allergy.lower() in text_lower and "allergy" not in text_lower:
                    issues.append(
                        f"Mentions {allergy} without noting patient allergy"
                    )

        # Severity assessment
        if not issues:
            severity = "none"
        elif any("critical" in i.lower() or "dangerous" in i.lower() for i in issues):
            severity = "critical"
        elif len(issues) >= 3:
            severity = "high"
        elif len(issues) >= 2:
            severity = "medium"
        else:
            severity = "low"

        return ClinicalSafetyResult(
            safe=len(issues) == 0,
            severity=severity,
            issues=issues,
        )
