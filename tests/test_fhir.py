"""Tests for FHIR R4 Interoperability Layer."""

import pytest
from fhir_layer.converter import FHIRConverter


class TestFHIRConverter:
    def test_parse_patient(self):
        resource = {
            "resourceType": "Patient",
            "id": "pt-001",
            "name": [{"given": ["John"], "family": "Doe"}],
            "gender": "male",
            "birthDate": "1960-05-15",
        }
        result = FHIRConverter.parse_patient(resource)
        assert result["patient_id"] == "pt-001"
        assert result["name"] == "John Doe"
        assert result["gender"] == "male"
        assert result["age"] > 0

    def test_parse_patient_invalid_type(self):
        with pytest.raises(ValueError, match="Expected Patient"):
            FHIRConverter.parse_patient({"resourceType": "Observation"})

    def test_parse_observations(self):
        observations = [
            {
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
                "valueQuantity": {"value": 95},
            },
            {
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "2708-6"}]},
                "valueQuantity": {"value": 97},
            },
        ]
        vitals = FHIRConverter.parse_observations(observations)
        assert vitals["heart_rate"] == 95
        assert vitals["oxygen_saturation"] == 97

    def test_parse_bp_panel(self):
        obs = {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9"}]},
            "component": [
                {
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": "8480-6"}]
                    },
                    "valueQuantity": {"value": 140},
                },
                {
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": "8462-4"}]
                    },
                    "valueQuantity": {"value": 90},
                },
            ],
        }
        vitals = FHIRConverter.parse_observations([obs])
        assert vitals["systolic_bp"] == 140
        assert vitals["diastolic_bp"] == 90

    def test_parse_bundle(self):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "pt-002",
                        "name": [{"given": ["Jane"], "family": "Smith"}],
                        "gender": "female",
                        "birthDate": "1975-01-01",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [{"system": "http://loinc.org", "code": "8867-4"}]
                        },
                        "valueQuantity": {"value": 88},
                    }
                },
            ],
        }
        result = FHIRConverter.parse_bundle(bundle)
        assert result["patient"]["patient_id"] == "pt-002"
        assert result["vitals"]["heart_rate"] == 88

    def test_to_risk_assessment(self):
        ra = FHIRConverter.to_risk_assessment(
            patient_id="pt-001",
            prediction="High Risk",
            risk_score=0.85,
            confidence=0.92,
            features={"heart_rate": 110, "age": 70},
        )
        assert ra["resourceType"] == "RiskAssessment"
        assert ra["status"] == "final"
        assert ra["subject"]["reference"] == "Patient/pt-001"
        assert len(ra["prediction"]) == 1

    def test_to_document_reference(self):
        dr = FHIRConverter.to_document_reference(
            patient_id="pt-001",
            original_note="Patient has chest pain...",
            summary="65yo with ACS",
        )
        assert dr["resourceType"] == "DocumentReference"
        assert dr["status"] == "current"
        assert len(dr["content"]) == 2

    def test_vitals_to_observations(self):
        vitals = {"heart_rate": 80, "oxygen_saturation": 98}
        obs = FHIRConverter.vitals_to_observations("pt-001", vitals)
        assert len(obs) == 2
        for o in obs:
            assert o["resourceType"] == "Observation"
            assert o["subject"]["reference"] == "Patient/pt-001"
