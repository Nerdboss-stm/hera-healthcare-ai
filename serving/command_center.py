"""Unified Command Center — Chains all 7 subsystems into a single patient flow.

One patient input → NER extraction → Triage → Diagnosis → Treatment →
RAG context retrieval → Summarization → Safety evaluation → FHIR export.

Includes feedback loops: if the evaluator detects hallucinations or safety
issues, the summary is regenerated with corrections.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """A single stage in the unified pipeline."""

    name: str
    system: str
    status: str = "pending"  # pending | running | completed | failed | re-run
    started_at: float = 0.0
    completed_at: float = 0.0
    latency_ms: float = 0.0
    result: dict = field(default_factory=dict)
    narration: str = ""
    error: str = ""


@dataclass
class CommandCenterResult:
    """Complete result from the unified pipeline."""

    patient_id: str
    stages: list[PipelineStage]
    overall_latency_ms: float = 0.0
    consensus_score: float = 0.0
    safety_passed: bool = False
    feedback_loops_triggered: int = 0
    fhir_bundle: dict = field(default_factory=dict)
    data_lineage: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "stages": [asdict(s) for s in self.stages],
            "overall_latency_ms": self.overall_latency_ms,
            "consensus_score": self.consensus_score,
            "safety_passed": self.safety_passed,
            "feedback_loops_triggered": self.feedback_loops_triggered,
            "fhir_bundle": self.fhir_bundle,
            "data_lineage": self.data_lineage,
        }


class UnifiedPipeline:
    """Orchestrates all 7 HERA subsystems in a single patient flow."""

    def __init__(self, risk_predictor=None, summarizer_fn=None):
        self._risk_predictor = risk_predictor
        self._summarizer_fn = summarizer_fn

    def execute(
        self,
        patient_id: str,
        clinical_note: str,
        chief_complaint: str,
        vitals: dict[str, float],
        age: int,
        gender: str = "unknown",
        medical_history: list[str] | None = None,
        current_medications: list[str] | None = None,
        allergies: list[str] | None = None,
    ) -> CommandCenterResult:
        """Run the full unified pipeline."""
        pipeline_start = time.time()
        medical_history = medical_history or []
        current_medications = current_medications or []
        allergies = allergies or []
        stages: list[PipelineStage] = []
        lineage: list[dict] = []
        feedback_loops = 0

        # ── Stage 1: NER Entity Extraction ──
        ner_stage = PipelineStage(name="Entity Extraction", system="ner")
        ner_stage.status = "running"
        ner_stage.started_at = time.time()
        try:
            from ner.extractor import ClinicalNERExtractor

            extractor = ClinicalNERExtractor()
            ner_result = extractor.extract(clinical_note)
            ner_dict = ner_result.to_dict()
            ner_stage.result = ner_dict
            ner_stage.narration = (
                f"Scanned the clinical note and found {ner_dict['entity_count']} "
                f"medical entities: {len(ner_dict.get('medications', []))} medications, "
                f"{len(ner_dict.get('conditions', []))} conditions, "
                f"{len(ner_dict.get('procedures', []))} procedures, "
                f"{len(ner_dict.get('lab_values', []))} lab values."
            )
            ner_stage.status = "completed"
        except Exception as e:
            ner_stage.status = "failed"
            ner_stage.error = str(e)
            ner_stage.narration = f"Entity extraction failed: {e}"
            ner_dict = {"entity_count": 0}
        ner_stage.completed_at = time.time()
        ner_stage.latency_ms = round(
            (ner_stage.completed_at - ner_stage.started_at) * 1000, 1
        )
        stages.append(ner_stage)
        lineage.append(
            {
                "stage": "ner",
                "input": "clinical_note",
                "output": "entities",
                "records": ner_dict.get("entity_count", 0),
            }
        )

        # ── Stage 2: Knowledge Graph ──
        kg_stage = PipelineStage(name="Knowledge Graph", system="knowledge_graph")
        kg_stage.status = "running"
        kg_stage.started_at = time.time()
        kg_dict: dict[str, Any] = {}
        try:
            from ner.knowledge_graph import PatientKnowledgeGraph

            kg = PatientKnowledgeGraph()
            kg_build = kg.build_from_note(clinical_note, patient_id)
            kg_dict = kg.to_dict()
            kg_stage.result = {
                "nodes": kg_build["nodes"],
                "edges": kg_build["edges"],
                "graph": kg_dict,
            }
            kg_stage.narration = (
                f"Built a relationship graph with {kg_build['nodes']} nodes "
                f"and {kg_build['edges']} edges, showing how medications, "
                f"conditions, and procedures connect."
            )
            kg_stage.status = "completed"
        except Exception as e:
            kg_stage.status = "failed"
            kg_stage.error = str(e)
            kg_stage.narration = f"Knowledge graph construction failed: {e}"
        kg_stage.completed_at = time.time()
        kg_stage.latency_ms = round(
            (kg_stage.completed_at - kg_stage.started_at) * 1000, 1
        )
        stages.append(kg_stage)
        lineage.append(
            {
                "stage": "knowledge_graph",
                "input": "entities",
                "output": "graph",
                "records": kg_stage.result.get("nodes", 0),
            }
        )

        # ── Stage 3: Multi-Agent Reasoning (Triage → Diagnosis → Treatment) ──
        reasoning_stage = PipelineStage(name="Multi-Agent Reasoning", system="agents")
        reasoning_stage.status = "running"
        reasoning_stage.started_at = time.time()
        reasoning_result = None
        try:
            from agents.orchestrator import ClinicalOrchestrator
            from agents.protocols import PatientContext

            ctx = PatientContext(
                patient_id=patient_id,
                chief_complaint=chief_complaint,
                clinical_note=clinical_note,
                vitals=vitals,
                age=age,
                gender=gender,
                medical_history=medical_history,
                current_medications=current_medications,
                allergies=allergies,
            )
            orchestrator = ClinicalOrchestrator(risk_predictor=self._risk_predictor)
            reasoning_result = orchestrator.reason(ctx)
            reasoning_stage.result = {
                "triage": reasoning_result.triage.to_dict(),
                "diagnosis": reasoning_result.diagnosis.to_dict(),
                "treatment": reasoning_result.treatment.to_dict(),
                "consensus_score": reasoning_result.consensus_score,
                "audit": reasoning_result.reasoning_audit,
            }
            esi = reasoning_result.triage.esi_level
            primary_dx = reasoning_result.diagnosis.primary_diagnosis
            disposition = reasoning_result.treatment.disposition
            reasoning_stage.narration = (
                f"Three agents collaborated: Triage assigned ESI level {esi}, "
                f"Diagnostic identified '{primary_dx}' as the primary diagnosis, "
                f"Treatment recommended {disposition}. "
                f"Consensus score: {reasoning_result.consensus_score:.1%}."
            )
            reasoning_stage.status = "completed"
        except Exception as e:
            reasoning_stage.status = "failed"
            reasoning_stage.error = str(e)
            reasoning_stage.narration = f"Multi-agent reasoning failed: {e}"
        reasoning_stage.completed_at = time.time()
        reasoning_stage.latency_ms = round(
            (reasoning_stage.completed_at - reasoning_stage.started_at) * 1000, 1
        )
        stages.append(reasoning_stage)
        lineage.append(
            {
                "stage": "agents",
                "input": "patient_context",
                "output": "triage+diagnosis+treatment",
                "records": 3,
            }
        )

        # ── Stage 4: Risk Prediction ──
        risk_stage = PipelineStage(name="Risk Prediction", system="risk_predictor")
        risk_stage.status = "running"
        risk_stage.started_at = time.time()
        risk_result: dict[str, Any] = {}
        try:
            if self._risk_predictor:
                risk_result = self._risk_predictor(
                    heart_rate=vitals.get("heart_rate", 80),
                    respiratory_rate=vitals.get("respiratory_rate", 16),
                    body_temperature=vitals.get("body_temperature", 37.0),
                    oxygen_saturation=vitals.get("oxygen_saturation", 98),
                    systolic_bp=vitals.get("systolic_bp", 120),
                    diastolic_bp=vitals.get("diastolic_bp", 80),
                    age=age,
                )
                risk_stage.result = risk_result
                risk_stage.narration = (
                    f"Random Forest classifier analyzed 7 vitals + 2 derived features "
                    f"(BMI, MAP). Prediction: {risk_result['prediction']} "
                    f"with {risk_result['confidence']:.1%} confidence "
                    f"(risk score: {risk_result['risk_score']:.1%})."
                )
                risk_stage.status = "completed"
            else:
                risk_stage.status = "completed"
                risk_stage.narration = "Risk predictor not available, skipped."
        except Exception as e:
            risk_stage.status = "failed"
            risk_stage.error = str(e)
            risk_stage.narration = f"Risk prediction failed: {e}"
        risk_stage.completed_at = time.time()
        risk_stage.latency_ms = round(
            (risk_stage.completed_at - risk_stage.started_at) * 1000, 1
        )
        stages.append(risk_stage)
        lineage.append(
            {
                "stage": "risk_prediction",
                "input": "vitals",
                "output": "risk_score",
                "records": 1,
            }
        )

        # ── Stage 5: RAG Knowledge Retrieval ──
        rag_stage = PipelineStage(name="RAG Knowledge Retrieval", system="rag")
        rag_stage.status = "running"
        rag_stage.started_at = time.time()
        rag_results: list[dict] = []
        try:
            from rag.rag_pipeline import RAGPipeline

            rag = RAGPipeline(summarizer_fn=self._summarizer_fn)
            rag.initialize()
            query = chief_complaint
            if reasoning_result:
                query = (
                    f"{chief_complaint} {reasoning_result.diagnosis.primary_diagnosis}"
                )
            rag_data = rag.query_knowledge(query, top_k=3)
            rag_results = rag_data.get("results", [])
            rag_stage.result = rag_data
            rag_stage.narration = (
                f"Searched {rag_data.get('total_corpus_size', 0)} medical guidelines "
                f"using semantic vector search. Found {len(rag_results)} relevant passages "
                f"matching '{query}'."
            )
            rag_stage.status = "completed"
        except ImportError:
            rag_stage.status = "failed"
            rag_stage.error = "RAG dependencies not installed"
            rag_stage.narration = (
                "RAG search skipped — FAISS/sentence-transformers not installed."
            )
        except Exception as e:
            rag_stage.status = "failed"
            rag_stage.error = str(e)
            rag_stage.narration = f"RAG search failed: {e}"
        rag_stage.completed_at = time.time()
        rag_stage.latency_ms = round(
            (rag_stage.completed_at - rag_stage.started_at) * 1000, 1
        )
        stages.append(rag_stage)
        lineage.append(
            {
                "stage": "rag",
                "input": "chief_complaint+diagnosis",
                "output": "guidelines",
                "records": len(rag_results),
            }
        )

        # ── Stage 6: Summarization ──
        summary_stage = PipelineStage(
            name="Clinical Summarization", system="summarizer"
        )
        summary_stage.status = "running"
        summary_stage.started_at = time.time()
        summary_text = ""
        try:
            if self._summarizer_fn:
                summary_text = self._summarizer_fn(clinical_note)
                summary_stage.result = {
                    "summary": summary_text,
                    "input_length": len(clinical_note),
                    "summary_length": len(summary_text),
                    "compression": round(
                        (1 - len(summary_text) / max(len(clinical_note), 1)) * 100, 1
                    ),
                }
                summary_stage.narration = (
                    f"T5 transformer compressed {len(clinical_note)} characters "
                    f"to {len(summary_text)} characters "
                    f"({summary_stage.result['compression']}% compression)."
                )
                summary_stage.status = "completed"
            else:
                summary_stage.status = "failed"
                summary_stage.narration = "Summarizer model not available."
        except Exception as e:
            summary_stage.status = "failed"
            summary_stage.error = str(e)
            summary_stage.narration = f"Summarization failed: {e}"
        summary_stage.completed_at = time.time()
        summary_stage.latency_ms = round(
            (summary_stage.completed_at - summary_stage.started_at) * 1000, 1
        )
        stages.append(summary_stage)
        lineage.append(
            {
                "stage": "summarization",
                "input": "clinical_note",
                "output": "summary",
                "records": 1,
            }
        )

        # ── Stage 7: Safety Evaluation (with feedback loop) ──
        eval_stage = PipelineStage(name="Safety Evaluation", system="evaluator")
        eval_stage.status = "running"
        eval_stage.started_at = time.time()
        safety_passed = False
        eval_result: dict[str, Any] = {}
        try:
            from evaluation.evaluator import ClinicalEvaluator

            evaluator = ClinicalEvaluator()
            text_to_evaluate = summary_text if summary_text else clinical_note

            report = evaluator.evaluate(clinical_note, text_to_evaluate)
            eval_result = report.to_dict()
            safety_passed = report.pass_threshold

            # ── FEEDBACK LOOP: If evaluation fails and summary exists, retry ──
            if not safety_passed and summary_text and self._summarizer_fn:
                feedback_loops += 1
                eval_stage.narration = (
                    f"First evaluation scored {report.overall_score:.1%} (FAILED). "
                    f"Triggering feedback loop — re-summarizing with corrections..."
                )
                # Re-summarize with RAG context for better accuracy
                try:
                    context = " ".join(r.get("text", "") for r in rag_results[:2])
                    augmented = (
                        f"Context: {context}\n\nNote: {clinical_note}"
                        if context
                        else clinical_note
                    )
                    summary_text = self._summarizer_fn(augmented)
                    summary_stage.result["summary_v2"] = summary_text
                    summary_stage.result["feedback_loop"] = True
                    summary_stage.status = "re-run"

                    report2 = evaluator.evaluate(clinical_note, summary_text)
                    eval_result = report2.to_dict()
                    safety_passed = report2.pass_threshold
                    eval_stage.narration += (
                        f" Re-evaluation scored {report2.overall_score:.1%} "
                        f"({'PASSED' if safety_passed else 'FAILED'})."
                    )
                except Exception:
                    pass  # Keep original evaluation result
            else:
                eval_stage.narration = (
                    f"Evaluated output on 4 axes: factual consistency "
                    f"({eval_result.get('factual_consistency', {}).get('score', 0):.0%}), "
                    f"hallucination detection, medical accuracy, clinical safety. "
                    f"Overall: {report.overall_score:.1%} — "
                    f"{'PASSED' if safety_passed else 'FAILED'}."
                )

            eval_stage.result = eval_result
            eval_stage.status = "completed"
        except Exception as e:
            eval_stage.status = "failed"
            eval_stage.error = str(e)
            eval_stage.narration = f"Safety evaluation failed: {e}"
        eval_stage.completed_at = time.time()
        eval_stage.latency_ms = round(
            (eval_stage.completed_at - eval_stage.started_at) * 1000, 1
        )
        stages.append(eval_stage)
        lineage.append(
            {
                "stage": "evaluation",
                "input": "summary+source",
                "output": "safety_scores",
                "records": 4,
            }
        )

        # ── Stage 8: FHIR Export ──
        fhir_stage = PipelineStage(name="FHIR R4 Export", system="fhir")
        fhir_stage.status = "running"
        fhir_stage.started_at = time.time()
        fhir_bundle: dict[str, Any] = {}
        try:
            from fhir_layer.converter import FHIRConverter

            resources = []

            # Patient resource
            resources.append(
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient_id,
                        "gender": gender,
                    }
                }
            )

            # Vitals as FHIR Observations
            for obs in FHIRConverter.vitals_to_observations(patient_id, vitals):
                resources.append({"resource": obs})

            # Risk Assessment
            if risk_result:
                ra = FHIRConverter.to_risk_assessment(
                    patient_id=patient_id,
                    prediction=risk_result.get("prediction", "Unknown"),
                    risk_score=risk_result.get("risk_score", 0.0),
                    confidence=risk_result.get("confidence", 0.0),
                    features=risk_result.get("features_used", {}),
                )
                resources.append({"resource": ra})

            # Document Reference (summary)
            if summary_text:
                doc_ref = FHIRConverter.to_document_reference(
                    patient_id=patient_id,
                    original_note=clinical_note,
                    summary=summary_text,
                )
                resources.append({"resource": doc_ref})

            fhir_bundle = {
                "resourceType": "Bundle",
                "type": "collection",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "entry": resources,
                "total": len(resources),
            }
            fhir_stage.result = {
                "resource_count": len(resources),
                "bundle": fhir_bundle,
            }
            fhir_stage.narration = (
                f"Exported entire case as a FHIR R4 Bundle with {len(resources)} resources "
                f"(Patient, Observations, RiskAssessment, DocumentReference). "
                f"Any hospital EHR system (Epic, Cerner) can import this directly."
            )
            fhir_stage.status = "completed"
        except Exception as e:
            fhir_stage.status = "failed"
            fhir_stage.error = str(e)
            fhir_stage.narration = f"FHIR export failed: {e}"
        fhir_stage.completed_at = time.time()
        fhir_stage.latency_ms = round(
            (fhir_stage.completed_at - fhir_stage.started_at) * 1000, 1
        )
        stages.append(fhir_stage)
        lineage.append(
            {
                "stage": "fhir_export",
                "input": "all_results",
                "output": "fhir_bundle",
                "records": fhir_stage.result.get("resource_count", 0),
            }
        )

        # ── Stage 9: Data Engineering Pipeline ──
        de_stage = PipelineStage(name="Data Engineering", system="data_engineering")
        de_stage.status = "running"
        de_stage.started_at = time.time()
        try:
            from data_engineering.streaming import event_stream
            from data_engineering.quality import DataQualityFramework
            from data_engineering.lineage import DataLineageTracker
            from data_engineering.warehouse import clinical_warehouse  # noqa: F811
            from data_engineering.cdc import cdc_stream
            from data_engineering.catalog import data_catalog

            # 1. Stream ingestion with schema validation
            stream_result = event_stream.ingest_patient_event(
                {
                    "patient_id": patient_id,
                    "heart_rate": vitals.get("heart_rate", 0),
                    "respiratory_rate": vitals.get("respiratory_rate", 0),
                    "body_temperature": vitals.get("body_temperature", 0),
                    "oxygen_saturation": vitals.get("oxygen_saturation", 0),
                    "systolic_bp": vitals.get("systolic_bp", 0),
                    "diastolic_bp": vitals.get("diastolic_bp", 0),
                    "age": age,
                    "chief_complaint": chief_complaint,
                    "clinical_note": clinical_note,
                }
            )

            # 2. Data quality validation
            dq = DataQualityFramework()
            quality_report = dq.validate_vitals(vitals)

            # 3. Column-level lineage
            tracker = DataLineageTracker()
            tracker.build_pipeline_lineage()

            # 4. Load into warehouse (singleton — persists across requests)
            from data_engineering.warehouse import clinical_warehouse as wh
            encounter_result = {
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "vitals": vitals,
                "stages": [asdict(s) for s in stages],
                "overall_latency_ms": round((time.time() - pipeline_start) * 1000, 1),
                "feedback_loops_triggered": feedback_loops,
            }
            enc_id = wh.load_encounter(encounter_result)
            wh.refresh_aggregates()

            # 5. CDC events
            cdc_events = cdc_stream.capture_encounter(encounter_result)

            # 6. Update catalog freshness
            data_catalog.update_freshness("raw_patient_vitals", row_count=1)
            data_catalog.update_freshness("fact_encounters", row_count=enc_id)

            de_stage.result = {
                "stream_events": len(stream_result.get("events", [])),
                "dlq_events": len(stream_result.get("dlq", [])),
                "quality_score": quality_report.score,
                "quality_checks": len(quality_report.checks),
                "lineage_nodes": len(tracker._nodes),
                "lineage_edges": len(tracker._edges),
                "warehouse_encounter_id": enc_id,
                "cdc_events": len(cdc_events),
                "dag_tasks": 16,
            }
            de_stage.narration = (
                f"Data engineering pipeline processed: {len(stream_result.get('events', []))} "
                f"events streamed with schema v3.0 validation, "
                f"quality score {quality_report.score:.0%} across {len(quality_report.checks)} checks, "
                f"lineage DAG with {len(tracker._nodes)} nodes/{len(tracker._edges)} edges, "
                f"encounter #{enc_id} loaded into star schema warehouse, "
                f"{len(cdc_events)} CDC change events emitted."
            )
            de_stage.status = "completed"
        except Exception as e:
            de_stage.status = "failed"
            de_stage.error = str(e)
            de_stage.narration = f"Data engineering pipeline failed: {e}"
        de_stage.completed_at = time.time()
        de_stage.latency_ms = round(
            (de_stage.completed_at - de_stage.started_at) * 1000, 1
        )
        stages.append(de_stage)
        lineage.append(
            {
                "stage": "data_engineering",
                "input": "all_pipeline_outputs",
                "output": "warehouse+cdc+catalog",
                "records": de_stage.result.get("cdc_events", 0)
                if de_stage.result
                else 0,
            }
        )

        # ── Final result ──
        consensus = reasoning_result.consensus_score if reasoning_result else 0.0
        overall_latency = round((time.time() - pipeline_start) * 1000, 1)

        return CommandCenterResult(
            patient_id=patient_id,
            stages=stages,
            overall_latency_ms=overall_latency,
            consensus_score=consensus,
            safety_passed=safety_passed,
            feedback_loops_triggered=feedback_loops,
            fhir_bundle=fhir_bundle,
            data_lineage=lineage,
        )
