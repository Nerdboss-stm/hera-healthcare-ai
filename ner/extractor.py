"""Clinical NER Extractor — Biomedical named entity recognition.

Extracts medications, conditions, procedures, lab values, dosages,
and anatomical entities from unstructured clinical notes using
rule-based patterns augmented by SciSpaCy when available.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Regex patterns for clinical entity extraction
MEDICATION_PATTERNS = [
    r'\b(?:aspirin|heparin|nitroglycerin|morphine|acetaminophen|ibuprofen|'
    r'metoprolol|lisinopril|atorvastatin|ceftriaxone|azithromycin|vancomycin|'
    r'piperacillin|furosemide|prednisone|albuterol|ipratropium|warfarin|'
    r'enoxaparin|rivaroxaban|amoxicillin|doxycycline|ciprofloxacin|'
    r'metformin|insulin|omeprazole|pantoprazole|ondansetron|lorazepam|'
    r'alteplase|norepinephrine|epinephrine|atropine|fentanyl|hydromorphone|'
    r'clopidogrel|ticagrelor|nitrofurantoin|metronidazole)\b',
]

CONDITION_PATTERNS = [
    r'\b(?:hypertension|diabetes|COPD|asthma|pneumonia|sepsis|stroke|'
    r'myocardial infarction|heart failure|atrial fibrillation|DVT|PE|'
    r'pulmonary embolism|appendicitis|cholecystitis|pancreatitis|'
    r'acute coronary syndrome|STEMI|NSTEMI|unstable angina|'
    r'subarachnoid hemorrhage|meningitis|cellulitis|UTI|'
    r'urinary tract infection|chronic kidney disease|cirrhosis|'
    r'anaphylaxis|status epilepticus|DKA|diabetic ketoacidosis)\b',
]

PROCEDURE_PATTERNS = [
    r'\b(?:intubation|ventilation|chest tube|central line|'
    r'lumbar puncture|thoracentesis|paracentesis|cardioversion|'
    r'CT scan|MRI|X-ray|ultrasound|ECG|EKG|echocardiogram|'
    r'colonoscopy|endoscopy|bronchoscopy|catheterization|PCI|'
    r'appendectomy|cholecystectomy|laparoscopy|biopsy|dialysis|'
    r'transfusion|CPR|defibrillation)\b',
]

LAB_VALUE_PATTERN = (
    r'(?:(?:troponin|BNP|NT-proBNP|WBC|hemoglobin|hematocrit|platelets|'
    r'creatinine|BUN|sodium|potassium|glucose|lactate|INR|PT|PTT|'
    r'D-dimer|procalcitonin|CRP|ESR|lipase|amylase|ALT|AST|bilirubin|'
    r'albumin|A1c|HbA1c|TSH|pH|pCO2|pO2|bicarbonate|SpO2)'
    r'\s*(?:of|=|:|\s)\s*'
    r'[\d]+\.?[\d]*\s*(?:mg/dL|mmol/L|mEq/L|g/dL|%|ng/mL|pg/mL|'
    r'U/L|IU/L|mcg/L|cells/uL|x10\^3|mmHg|sec)?)'
)

DOSAGE_PATTERN = r'(\d+\.?\d*)\s*(mg|mcg|g|mL|units?|mEq)\b'

VITAL_SIGN_PATTERN = (
    r'(?:(?:HR|heart rate|pulse)\s*(?:of|=|:|\s)\s*(\d+))|'
    r'(?:(?:BP|blood pressure)\s*(?:of|=|:|\s)\s*(\d+/\d+))|'
    r'(?:(?:RR|respiratory rate)\s*(?:of|=|:|\s)\s*(\d+))|'
    r'(?:(?:temp|temperature)\s*(?:of|=|:|\s)\s*([\d.]+))|'
    r'(?:(?:SpO2|O2 sat|oxygen saturation)\s*(?:of|=|:|\s)\s*(\d+))'
)


@dataclass
class ClinicalEntity:
    """A single extracted clinical entity."""
    text: str
    label: str  # MEDICATION, CONDITION, PROCEDURE, LAB_VALUE, VITAL_SIGN, DOSAGE
    start: int
    end: int
    confidence: float = 1.0
    normalized: str = ""  # UMLS/SNOMED normalized form
    code: str = ""        # UMLS CUI or SNOMED code


@dataclass
class ExtractionResult:
    """Complete NER extraction output."""
    entities: list[ClinicalEntity]
    medications: list[ClinicalEntity]
    conditions: list[ClinicalEntity]
    procedures: list[ClinicalEntity]
    lab_values: list[ClinicalEntity]
    vital_signs: list[ClinicalEntity]
    entity_count: int = 0

    def to_dict(self) -> dict:
        return {
            "entity_count": self.entity_count,
            "medications": [{"text": e.text, "label": e.label} for e in self.medications],
            "conditions": [{"text": e.text, "label": e.label} for e in self.conditions],
            "procedures": [{"text": e.text, "label": e.label} for e in self.procedures],
            "lab_values": [{"text": e.text, "label": e.label} for e in self.lab_values],
            "vital_signs": [{"text": e.text, "label": e.label} for e in self.vital_signs],
        }


# Simplified UMLS concept mapping (subset)
UMLS_MAP = {
    "aspirin": ("C0004057", "Aspirin"),
    "heparin": ("C0019134", "Heparin"),
    "morphine": ("C0026549", "Morphine"),
    "hypertension": ("C0020538", "Hypertensive disease"),
    "diabetes": ("C0011849", "Diabetes mellitus"),
    "pneumonia": ("C0032285", "Pneumonia"),
    "sepsis": ("C0036690", "Sepsis"),
    "stroke": ("C0038454", "Cerebrovascular accident"),
    "myocardial infarction": ("C0027051", "Myocardial infarction"),
    "heart failure": ("C0018801", "Heart failure"),
    "copd": ("C0024117", "COPD"),
    "pulmonary embolism": ("C0034065", "Pulmonary embolism"),
    "appendicitis": ("C0003615", "Appendicitis"),
    "atrial fibrillation": ("C0004238", "Atrial fibrillation"),
}


class ClinicalNERExtractor:
    """Extracts structured medical entities from clinical notes.

    Uses regex-based patterns for reliable extraction, with optional
    SciSpaCy biomedical model augmentation when available.
    """

    def __init__(self, use_scispacy: bool = False):
        self._nlp = None
        if use_scispacy:
            try:
                import spacy
                self._nlp = spacy.load("en_ner_bc5cdr_md")
                logger.info("SciSpaCy model loaded for NER augmentation")
            except (ImportError, OSError):
                logger.info("SciSpaCy not available, using regex-only extraction")

    def extract(self, text: str) -> ExtractionResult:
        """Extract all clinical entities from text."""
        entities = []
        entities.extend(self._extract_medications(text))
        entities.extend(self._extract_conditions(text))
        entities.extend(self._extract_procedures(text))
        entities.extend(self._extract_lab_values(text))
        entities.extend(self._extract_vital_signs(text))

        # Augment with SciSpaCy if available
        if self._nlp:
            entities.extend(self._scispacy_extract(text))

        # Deduplicate by (text, label) pair
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in seen:
                seen.add(key)
                self._normalize_entity(e)
                unique.append(e)

        medications = [e for e in unique if e.label == "MEDICATION"]
        conditions = [e for e in unique if e.label == "CONDITION"]
        procedures = [e for e in unique if e.label == "PROCEDURE"]
        lab_values = [e for e in unique if e.label == "LAB_VALUE"]
        vital_signs = [e for e in unique if e.label == "VITAL_SIGN"]

        return ExtractionResult(
            entities=unique,
            medications=medications,
            conditions=conditions,
            procedures=procedures,
            lab_values=lab_values,
            vital_signs=vital_signs,
            entity_count=len(unique),
        )

    def _extract_medications(self, text: str) -> list[ClinicalEntity]:
        entities = []
        for pattern in MEDICATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ClinicalEntity(
                    text=match.group(),
                    label="MEDICATION",
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _extract_conditions(self, text: str) -> list[ClinicalEntity]:
        entities = []
        for pattern in CONDITION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ClinicalEntity(
                    text=match.group(),
                    label="CONDITION",
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _extract_procedures(self, text: str) -> list[ClinicalEntity]:
        entities = []
        for pattern in PROCEDURE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ClinicalEntity(
                    text=match.group(),
                    label="PROCEDURE",
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _extract_lab_values(self, text: str) -> list[ClinicalEntity]:
        entities = []
        for match in re.finditer(LAB_VALUE_PATTERN, text, re.IGNORECASE):
            entities.append(ClinicalEntity(
                text=match.group().strip(),
                label="LAB_VALUE",
                start=match.start(),
                end=match.end(),
            ))
        return entities

    def _extract_vital_signs(self, text: str) -> list[ClinicalEntity]:
        entities = []
        for match in re.finditer(VITAL_SIGN_PATTERN, text, re.IGNORECASE):
            entities.append(ClinicalEntity(
                text=match.group().strip(),
                label="VITAL_SIGN",
                start=match.start(),
                end=match.end(),
            ))
        return entities

    def _scispacy_extract(self, text: str) -> list[ClinicalEntity]:
        """Augment extraction with SciSpaCy biomedical NER."""
        entities = []
        doc = self._nlp(text)
        label_map = {
            "CHEMICAL": "MEDICATION",
            "DISEASE": "CONDITION",
        }
        for ent in doc.ents:
            mapped_label = label_map.get(ent.label_, ent.label_)
            if mapped_label in ("MEDICATION", "CONDITION"):
                entities.append(ClinicalEntity(
                    text=ent.text,
                    label=mapped_label,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9,
                ))
        return entities

    def _normalize_entity(self, entity: ClinicalEntity) -> None:
        """Map entity to UMLS concept if available."""
        key = entity.text.lower()
        if key in UMLS_MAP:
            entity.code, entity.normalized = UMLS_MAP[key]
