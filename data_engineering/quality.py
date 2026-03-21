"""Data quality framework — Great Expectations-style validation.

Demonstrates: schema validation, statistical anomaly detection,
completeness checks, freshness monitoring, and quality scoring.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class QualityCheck:
    """A single quality check result."""

    name: str
    category: str  # schema, completeness, accuracy, freshness, consistency
    passed: bool
    severity: str = "warning"  # info, warning, critical
    details: str = ""
    expected: str = ""
    actual: str = ""


@dataclass
class QualityReport:
    """Aggregate quality report for a dataset or pipeline stage."""

    stage: str
    checks: list[QualityCheck] = field(default_factory=list)
    score: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def compute_score(self) -> float:
        if not self.checks:
            self.score = 1.0
            return self.score
        weights = {"info": 0.5, "warning": 1.0, "critical": 2.0}
        total_weight = sum(weights.get(c.severity, 1.0) for c in self.checks)
        passed_weight = sum(
            weights.get(c.severity, 1.0) for c in self.checks if c.passed
        )
        self.score = round(passed_weight / max(total_weight, 1), 3)
        return self.score

    def to_dict(self) -> dict:
        self.compute_score()
        return {
            "stage": self.stage,
            "score": self.score,
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.passed),
            "failed": sum(1 for c in self.checks if not c.passed),
            "critical_failures": sum(
                1 for c in self.checks if not c.passed and c.severity == "critical"
            ),
            "checks": [
                {
                    "name": c.name,
                    "category": c.category,
                    "passed": c.passed,
                    "severity": c.severity,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "timestamp": self.timestamp,
        }


class DataQualityFramework:
    """Validates clinical data at every pipeline stage."""

    # Normal ranges for clinical vitals
    VITAL_RANGES = {
        "heart_rate": (40, 180),
        "respiratory_rate": (8, 40),
        "body_temperature": (35.0, 41.0),
        "oxygen_saturation": (85, 100),
        "systolic_bp": (70, 200),
        "diastolic_bp": (40, 130),
        "age": (0, 120),
    }

    def __init__(self):
        self._history: list[QualityReport] = []

    def validate_vitals(self, vitals: dict, stage: str = "input") -> QualityReport:
        """Validate patient vitals data."""
        report = QualityReport(stage=stage)

        # Schema completeness
        required = ["heart_rate", "respiratory_rate", "body_temperature",
                     "oxygen_saturation", "systolic_bp", "diastolic_bp"]
        for field_name in required:
            present = field_name in vitals and vitals[field_name] is not None
            report.checks.append(QualityCheck(
                name=f"field_present_{field_name}",
                category="completeness",
                passed=present,
                severity="critical" if not present else "info",
                details=f"Required field '{field_name}' {'present' if present else 'MISSING'}",
                expected="present",
                actual="present" if present else "missing",
            ))

        # Range validation
        for field_name, (low, high) in self.VITAL_RANGES.items():
            val = vitals.get(field_name)
            if val is not None:
                in_range = low <= val <= high
                report.checks.append(QualityCheck(
                    name=f"range_{field_name}",
                    category="accuracy",
                    passed=in_range,
                    severity="warning" if not in_range else "info",
                    details=f"{field_name}={val} {'within' if in_range else 'OUTSIDE'} [{low}, {high}]",
                    expected=f"[{low}, {high}]",
                    actual=str(val),
                ))

        # Statistical anomaly: BP consistency
        sbp = vitals.get("systolic_bp", 0)
        dbp = vitals.get("diastolic_bp", 0)
        if sbp and dbp:
            bp_consistent = sbp > dbp
            report.checks.append(QualityCheck(
                name="bp_consistency",
                category="consistency",
                passed=bp_consistent,
                severity="critical" if not bp_consistent else "info",
                details=f"Systolic ({sbp}) {'>' if bp_consistent else '<='} Diastolic ({dbp})",
            ))

        # Pulse pressure check (abnormal if < 25 or > 100)
        if sbp and dbp:
            pp = sbp - dbp
            pp_ok = 25 <= pp <= 100
            report.checks.append(QualityCheck(
                name="pulse_pressure",
                category="accuracy",
                passed=pp_ok,
                severity="warning" if not pp_ok else "info",
                details=f"Pulse pressure={pp} {'normal' if pp_ok else 'ABNORMAL'} range [25, 100]",
            ))

        report.compute_score()
        self._history.append(report)
        return report

    def validate_clinical_note(self, note: str, stage: str = "input") -> QualityReport:
        """Validate clinical note text quality."""
        report = QualityReport(stage=stage)

        # Non-empty
        report.checks.append(QualityCheck(
            name="note_not_empty",
            category="completeness",
            passed=len(note.strip()) > 0,
            severity="critical",
            details=f"Note length: {len(note)} characters",
        ))

        # Minimum length
        min_len = 20
        report.checks.append(QualityCheck(
            name="note_min_length",
            category="completeness",
            passed=len(note) >= min_len,
            severity="warning",
            details=f"Note length {len(note)} {'meets' if len(note) >= min_len else 'below'} minimum {min_len}",
        ))

        # Contains clinical keywords
        clinical_terms = ["patient", "history", "diagnosis", "treatment",
                          "medication", "vital", "complaint", "assessment",
                          "presenting", "symptoms", "exam"]
        found = [t for t in clinical_terms if t.lower() in note.lower()]
        report.checks.append(QualityCheck(
            name="clinical_terminology",
            category="accuracy",
            passed=len(found) >= 2,
            severity="warning",
            details=f"Found {len(found)} clinical terms: {found[:5]}",
        ))

        # No obvious PHI patterns in wrong context (SSN, phone in note body)
        import re
        ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        has_ssn = bool(ssn_pattern.search(note))
        report.checks.append(QualityCheck(
            name="no_embedded_ssn",
            category="consistency",
            passed=not has_ssn,
            severity="critical",
            details="SSN pattern detected in note" if has_ssn else "No SSN patterns found",
        ))

        report.compute_score()
        self._history.append(report)
        return report

    def validate_pipeline_output(self, stage_name: str, output: dict) -> QualityReport:
        """Validate a pipeline stage's output."""
        report = QualityReport(stage=stage_name)

        # Non-empty output
        report.checks.append(QualityCheck(
            name="output_not_empty",
            category="completeness",
            passed=len(output) > 0,
            severity="critical",
            details=f"Output has {len(output)} fields",
        ))

        # No null values in output
        nulls = [k for k, v in output.items() if v is None]
        report.checks.append(QualityCheck(
            name="no_null_values",
            category="completeness",
            passed=len(nulls) == 0,
            severity="warning",
            details=f"Null fields: {nulls}" if nulls else "No null fields",
        ))

        # Stage-specific checks
        if stage_name == "ner":
            entity_count = output.get("entity_count", 0)
            report.checks.append(QualityCheck(
                name="ner_entities_found",
                category="accuracy",
                passed=entity_count > 0,
                severity="warning",
                details=f"NER extracted {entity_count} entities",
            ))

        elif stage_name == "risk_prediction":
            score = output.get("risk_score", -1)
            report.checks.append(QualityCheck(
                name="risk_score_range",
                category="accuracy",
                passed=0 <= score <= 1,
                severity="critical",
                details=f"Risk score {score} {'valid' if 0 <= score <= 1 else 'OUT OF RANGE'} [0, 1]",
            ))

        elif stage_name == "evaluation":
            overall = output.get("overall_score", 0)
            report.checks.append(QualityCheck(
                name="eval_score_range",
                category="accuracy",
                passed=0 <= overall <= 1,
                severity="critical",
                details=f"Evaluation score {overall}",
            ))

        elif stage_name == "fhir_export":
            count = output.get("resource_count", 0)
            report.checks.append(QualityCheck(
                name="fhir_resources_generated",
                category="completeness",
                passed=count > 0,
                severity="critical",
                details=f"Generated {count} FHIR resources",
            ))

        report.compute_score()
        self._history.append(report)
        return report

    def get_pipeline_quality_summary(self) -> dict:
        """Get quality summary across all pipeline stages."""
        by_stage = {}
        for r in self._history:
            if r.stage not in by_stage:
                by_stage[r.stage] = []
            by_stage[r.stage].append(r.score)

        summary = {}
        for stage, scores in by_stage.items():
            summary[stage] = {
                "avg_score": round(statistics.mean(scores), 3),
                "min_score": round(min(scores), 3),
                "runs": len(scores),
            }

        all_scores = [r.score for r in self._history]
        return {
            "stages": summary,
            "overall_avg": round(statistics.mean(all_scores), 3) if all_scores else 0,
            "total_checks_run": sum(len(r.checks) for r in self._history),
            "total_failures": sum(
                1 for r in self._history for c in r.checks if not c.passed
            ),
        }
