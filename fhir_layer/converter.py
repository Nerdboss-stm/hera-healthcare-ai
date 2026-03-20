"""FHIR R4 Converter — Bidirectional conversion between FHIR and internal formats.

Parses FHIR Patient, Observation, and Bundle resources into Hera's internal
formats, and converts prediction/summarization outputs back into
FHIR-compliant RiskAssessment and DocumentReference resources.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)


class FHIRConverter:
    """Converts between HL7 FHIR R4 resources and Hera internal formats."""

    # FHIR Observation code → internal vital sign name
    LOINC_TO_VITAL = {
        "8867-4": "heart_rate",
        "9279-1": "respiratory_rate",
        "8310-5": "body_temperature",
        "2708-6": "oxygen_saturation",
        "8480-6": "systolic_bp",
        "8462-4": "diastolic_bp",
        "30525-0": "age",
        "85354-9": "blood_pressure",  # BP panel
    }

    VITAL_TO_LOINC = {v: k for k, v in LOINC_TO_VITAL.items() if v != "blood_pressure"}

    @staticmethod
    def parse_patient(resource: dict) -> dict:
        """Extract patient demographics from a FHIR Patient resource."""
        if resource.get("resourceType") != "Patient":
            raise ValueError(
                f"Expected Patient resource, got {resource.get('resourceType')}"
            )

        name_parts = []
        for name in resource.get("name", []):
            given = " ".join(name.get("given", []))
            family = name.get("family", "")
            name_parts.append(f"{given} {family}".strip())

        birth_date = resource.get("birthDate", "")
        age = 0
        if birth_date:
            try:
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                age = (datetime.now() - birth).days // 365
            except ValueError:
                pass

        return {
            "patient_id": resource.get("id", str(uuid4())),
            "name": name_parts[0] if name_parts else "Unknown",
            "gender": resource.get("gender", "unknown"),
            "age": age,
            "birth_date": birth_date,
        }

    @classmethod
    def parse_observations(cls, observations: list[dict]) -> dict:
        """Extract vital signs from a list of FHIR Observation resources."""
        vitals = {}
        for obs in observations:
            if obs.get("resourceType") != "Observation":
                continue

            # Get LOINC code
            code = None
            for coding in obs.get("code", {}).get("coding", []):
                if coding.get("system") == "http://loinc.org":
                    code = coding.get("code")
                    break

            if not code:
                continue

            # Handle BP panel (compound observation)
            if code == "85354-9":
                for component in obs.get("component", []):
                    comp_code = None
                    for coding in component.get("code", {}).get("coding", []):
                        if coding.get("system") == "http://loinc.org":
                            comp_code = coding.get("code")
                    if comp_code and comp_code in cls.LOINC_TO_VITAL:
                        vital_name = cls.LOINC_TO_VITAL[comp_code]
                        value = component.get("valueQuantity", {}).get("value")
                        if value is not None:
                            vitals[vital_name] = float(value)
                continue

            vital_name = cls.LOINC_TO_VITAL.get(code)
            if vital_name:
                value = obs.get("valueQuantity", {}).get("value")
                if value is not None:
                    vitals[vital_name] = float(value)

        return vitals

    @classmethod
    def parse_bundle(cls, bundle: dict) -> dict:
        """Parse a FHIR Bundle containing Patient + Observations."""
        if bundle.get("resourceType") != "Bundle":
            raise ValueError(
                f"Expected Bundle resource, got {bundle.get('resourceType')}"
            )

        patient_data = {}
        observations = []

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            rt = resource.get("resourceType")
            if rt == "Patient":
                patient_data = cls.parse_patient(resource)
            elif rt == "Observation":
                observations.append(resource)

        vitals = cls.parse_observations(observations)

        return {
            "patient": patient_data,
            "vitals": vitals,
        }

    @classmethod
    def to_risk_assessment(
        cls,
        patient_id: str,
        prediction: str,
        risk_score: float,
        confidence: float,
        features: dict,
    ) -> dict:
        """Convert a risk prediction result to a FHIR RiskAssessment resource."""
        return {
            "resourceType": "RiskAssessment",
            "id": str(uuid4()),
            "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "occurrenceDateTime": datetime.now(timezone.utc).isoformat(),
            "method": {
                "coding": [
                    {
                        "system": "http://hera-healthcare.ai/methods",
                        "code": "rf-risk-v1",
                        "display": "Random Forest Risk Prediction Model v1",
                    }
                ]
            },
            "prediction": [
                {
                    "outcome": {
                        "coding": [
                            {
                                "system": "http://hera-healthcare.ai/outcomes",
                                "code": "high-risk"
                                if prediction == "High Risk"
                                else "low-risk",
                                "display": prediction,
                            }
                        ]
                    },
                    "probabilityDecimal": risk_score,
                    "qualitativeRisk": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                                "code": "high"
                                if risk_score > 0.7
                                else "moderate"
                                if risk_score > 0.4
                                else "low",
                            }
                        ]
                    },
                }
            ],
            "basis": [{"display": f"{k}: {v}"} for k, v in features.items()],
            "meta": {
                "profile": ["http://hl7.org/fhir/StructureDefinition/RiskAssessment"],
                "tag": [
                    {"code": "hera-ai", "display": "Generated by HERA Healthcare AI"}
                ],
            },
        }

    @classmethod
    def to_document_reference(
        cls,
        patient_id: str,
        original_note: str,
        summary: str,
    ) -> dict:
        """Convert a clinical summary to a FHIR DocumentReference resource."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "resourceType": "DocumentReference",
            "id": str(uuid4()),
            "status": "current",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "34133-9",
                        "display": "Summary of episode note",
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "date": now,
            "author": [
                {
                    "display": "HERA Clinical Summarizer (T5-small fine-tuned)",
                }
            ],
            "description": "AI-generated clinical note summary",
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": summary,
                        "title": "Clinical Summary",
                        "creation": now,
                    }
                },
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": original_note,
                        "title": "Original Clinical Note",
                        "creation": now,
                    }
                },
            ],
            "meta": {
                "profile": [
                    "http://hl7.org/fhir/StructureDefinition/DocumentReference"
                ],
                "tag": [
                    {"code": "hera-ai", "display": "Generated by HERA Healthcare AI"}
                ],
            },
        }

    @classmethod
    def vitals_to_observations(cls, patient_id: str, vitals: dict) -> list[dict]:
        """Convert internal vitals dict to FHIR Observation resources."""
        observations = []
        units = {
            "heart_rate": ("/min", "beats/minute"),
            "respiratory_rate": ("/min", "breaths/minute"),
            "body_temperature": ("Cel", "degrees Celsius"),
            "oxygen_saturation": ("%", "percent"),
            "systolic_bp": ("mm[Hg]", "mmHg"),
            "diastolic_bp": ("mm[Hg]", "mmHg"),
            "age": ("a", "years"),
        }

        for vital_name, value in vitals.items():
            loinc = cls.VITAL_TO_LOINC.get(vital_name)
            if not loinc:
                continue

            unit_code, unit_display = units.get(vital_name, ("", ""))
            observations.append(
                {
                    "resourceType": "Observation",
                    "id": str(uuid4()),
                    "status": "final",
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": loinc,
                                "display": vital_name.replace("_", " ").title(),
                            }
                        ]
                    },
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "valueQuantity": {
                        "value": value,
                        "unit": unit_display,
                        "system": "http://unitsofmeasure.org",
                        "code": unit_code,
                    },
                    "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
                }
            )

        return observations
