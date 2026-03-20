"""Multi-Agent Clinical Reasoning Pipeline.

Implements a collaborative AI system where specialized agents
(Triage → Diagnostic → Treatment) work together on patient cases,
mimicking real clinical team workflows.
"""

from agents.orchestrator import ClinicalOrchestrator
from agents.triage import TriageAgent
from agents.diagnostic import DiagnosticAgent
from agents.treatment import TreatmentAgent

__all__ = [
    "ClinicalOrchestrator",
    "TriageAgent",
    "DiagnosticAgent",
    "TreatmentAgent",
]
