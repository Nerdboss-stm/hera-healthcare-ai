"""Column-level data lineage DAG — tracks every field transformation.

Demonstrates: field-level provenance, transformation DAG, impact analysis,
upstream/downstream tracing, and HIPAA-compliant data governance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class ColumnLineage:
    """Tracks a single column's journey through the pipeline."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    transformation: str  # e.g., "direct_copy", "derived", "aggregated", "normalized"
    expression: str = ""  # e.g., "dbp + (sbp - dbp) / 3"
    stage: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class LineageNode:
    """A node in the lineage DAG (table.column)."""

    table: str
    column: str
    dtype: str = "unknown"
    pii: bool = False
    description: str = ""

    @property
    def fqn(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class LineageEdge:
    """An edge in the lineage DAG."""

    source: str  # fqn
    target: str  # fqn
    transformation: str
    expression: str = ""
    stage: str = ""


class DataLineageTracker:
    """Full column-level lineage tracking across the HERA pipeline."""

    def __init__(self):
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge] = []
        self._stage_lineages: dict[str, list[ColumnLineage]] = {}

    def add_node(self, table: str, column: str, dtype: str = "unknown",
                 pii: bool = False, description: str = "") -> LineageNode:
        node = LineageNode(table=table, column=column, dtype=dtype,
                           pii=pii, description=description)
        self._nodes[node.fqn] = node
        return node

    def add_edge(self, source_table: str, source_col: str,
                 target_table: str, target_col: str,
                 transformation: str, expression: str = "",
                 stage: str = "") -> LineageEdge:
        src = f"{source_table}.{source_col}"
        tgt = f"{target_table}.{target_col}"
        # Ensure nodes exist
        if src not in self._nodes:
            self.add_node(source_table, source_col)
        if tgt not in self._nodes:
            self.add_node(target_table, target_col)

        edge = LineageEdge(source=src, target=tgt, transformation=transformation,
                           expression=expression, stage=stage)
        self._edges.append(edge)

        col = ColumnLineage(
            source_table=source_table, source_column=source_col,
            target_table=target_table, target_column=target_col,
            transformation=transformation, expression=expression, stage=stage,
        )
        self._stage_lineages.setdefault(stage, []).append(col)
        return edge

    def get_upstream(self, table: str, column: str) -> list[dict]:
        """Trace all upstream sources for a column."""
        fqn = f"{table}.{column}"
        visited = set()
        result = []

        def _trace(target: str, depth: int = 0):
            if target in visited:
                return
            visited.add(target)
            for e in self._edges:
                if e.target == target:
                    result.append({
                        "source": e.source,
                        "target": e.target,
                        "transformation": e.transformation,
                        "expression": e.expression,
                        "stage": e.stage,
                        "depth": depth,
                    })
                    _trace(e.source, depth + 1)

        _trace(fqn)
        return result

    def get_downstream(self, table: str, column: str) -> list[dict]:
        """Trace all downstream consumers of a column."""
        fqn = f"{table}.{column}"
        visited = set()
        result = []

        def _trace(source: str, depth: int = 0):
            if source in visited:
                return
            visited.add(source)
            for e in self._edges:
                if e.source == source:
                    result.append({
                        "source": e.source,
                        "target": e.target,
                        "transformation": e.transformation,
                        "expression": e.expression,
                        "stage": e.stage,
                        "depth": depth,
                    })
                    _trace(e.target, depth + 1)

        _trace(fqn)
        return result

    def get_pii_columns(self) -> list[dict]:
        """List all PII-flagged columns for HIPAA compliance."""
        return [
            {"fqn": n.fqn, "dtype": n.dtype, "description": n.description}
            for n in self._nodes.values() if n.pii
        ]

    def impact_analysis(self, table: str, column: str) -> dict:
        """If this column changes, what breaks downstream?"""
        downstream = self.get_downstream(table, column)
        affected_stages = set(d["stage"] for d in downstream if d["stage"])
        affected_columns = set(d["target"] for d in downstream)
        return {
            "source": f"{table}.{column}",
            "affected_columns": sorted(affected_columns),
            "affected_stages": sorted(affected_stages),
            "total_downstream": len(downstream),
        }

    def build_pipeline_lineage(self) -> None:
        """Build the full lineage DAG for the HERA clinical pipeline."""
        # Input layer — raw patient data
        for col in ["patient_id", "heart_rate", "respiratory_rate",
                     "body_temperature", "oxygen_saturation",
                     "systolic_bp", "diastolic_bp", "age", "clinical_note"]:
            pii = col in ("patient_id", "clinical_note")
            self.add_node("raw_input", col, dtype="str" if col in ("patient_id", "clinical_note") else "float",
                          pii=pii, description=f"Raw patient {col}")

        # Stage 1: NER extraction
        for entity_type in ["medications", "conditions", "procedures", "lab_values", "vital_signs"]:
            self.add_node("ner_output", entity_type, dtype="list[dict]",
                          pii=True, description=f"Extracted {entity_type}")
            self.add_edge("raw_input", "clinical_note", "ner_output", entity_type,
                          "regex_extraction", f"pattern_match(clinical_note, '{entity_type}')",
                          stage="ner")

        # Stage 2: Knowledge Graph
        for kg_field in ["nodes", "edges", "graph_data"]:
            self.add_node("knowledge_graph", kg_field, dtype="dict")
            self.add_edge("ner_output", "medications", "knowledge_graph", kg_field,
                          "graph_construction", "build_relationships(entities)", stage="knowledge_graph")
            self.add_edge("ner_output", "conditions", "knowledge_graph", kg_field,
                          "graph_construction", "build_relationships(entities)", stage="knowledge_graph")

        # Stage 3: Multi-Agent Reasoning
        for field in ["esi_level", "primary_diagnosis", "disposition"]:
            self.add_node("reasoning_output", field, dtype="str")
        self.add_edge("raw_input", "heart_rate", "reasoning_output", "esi_level",
                      "triage_classification", "triage_agent(vitals, complaint)", stage="agents")
        self.add_edge("raw_input", "clinical_note", "reasoning_output", "primary_diagnosis",
                      "diagnostic_inference", "diagnostic_agent(note, vitals)", stage="agents")
        self.add_edge("reasoning_output", "primary_diagnosis", "reasoning_output", "disposition",
                      "treatment_planning", "treatment_agent(diagnosis, history)", stage="agents")

        # Stage 4: Risk Prediction — derived features
        self.add_node("derived_features", "mean_arterial_pressure", dtype="float",
                      description="MAP = dbp + (sbp - dbp) / 3")
        self.add_edge("raw_input", "systolic_bp", "derived_features", "mean_arterial_pressure",
                      "derived", "dbp + (sbp - dbp) / 3", stage="risk_prediction")
        self.add_edge("raw_input", "diastolic_bp", "derived_features", "mean_arterial_pressure",
                      "derived", "dbp + (sbp - dbp) / 3", stage="risk_prediction")

        self.add_node("risk_output", "risk_score", dtype="float")
        self.add_node("risk_output", "prediction", dtype="str")
        self.add_node("risk_output", "confidence", dtype="float")
        for vital in ["heart_rate", "respiratory_rate", "body_temperature",
                       "oxygen_saturation", "age"]:
            self.add_edge("raw_input", vital, "risk_output", "risk_score",
                          "ml_inference", "random_forest.predict(features)", stage="risk_prediction")
        self.add_edge("derived_features", "mean_arterial_pressure", "risk_output", "risk_score",
                      "ml_inference", "random_forest.predict(features)", stage="risk_prediction")

        # Stage 5: RAG
        self.add_node("rag_output", "retrieved_passages", dtype="list[dict]")
        self.add_node("rag_output", "similarity_scores", dtype="list[float]")
        self.add_edge("raw_input", "clinical_note", "rag_output", "retrieved_passages",
                      "vector_search", "faiss.search(embed(note), top_k=3)", stage="rag")
        self.add_edge("reasoning_output", "primary_diagnosis", "rag_output", "retrieved_passages",
                      "vector_search", "faiss.search(embed(diagnosis), top_k=3)", stage="rag")

        # Stage 6: Summarization
        self.add_node("summary_output", "summary_text", dtype="str", pii=True)
        self.add_node("summary_output", "compression_ratio", dtype="float")
        self.add_edge("raw_input", "clinical_note", "summary_output", "summary_text",
                      "t5_generation", "t5_model.generate(note)", stage="summarization")
        self.add_edge("rag_output", "retrieved_passages", "summary_output", "summary_text",
                      "context_augmentation", "augment(note, rag_context)", stage="summarization")

        # Stage 7: Safety Evaluation
        for metric in ["factual_consistency", "hallucination_score",
                        "medical_accuracy", "clinical_safety"]:
            self.add_node("eval_output", metric, dtype="float")
            self.add_edge("summary_output", "summary_text", "eval_output", metric,
                          "llm_judge", f"evaluate_{metric}(source, summary)", stage="evaluation")
            self.add_edge("raw_input", "clinical_note", "eval_output", metric,
                          "llm_judge", f"evaluate_{metric}(source, summary)", stage="evaluation")

        # Stage 8: FHIR Export
        for resource in ["Patient", "Observation", "RiskAssessment", "DocumentReference"]:
            self.add_node("fhir_output", resource, dtype="dict")
        self.add_edge("raw_input", "patient_id", "fhir_output", "Patient",
                      "fhir_mapping", "map_to_fhir_patient(id, gender)", stage="fhir_export")
        self.add_edge("raw_input", "heart_rate", "fhir_output", "Observation",
                      "fhir_mapping", "vitals_to_observations(vitals)", stage="fhir_export")
        self.add_edge("risk_output", "risk_score", "fhir_output", "RiskAssessment",
                      "fhir_mapping", "to_risk_assessment(score)", stage="fhir_export")
        self.add_edge("summary_output", "summary_text", "fhir_output", "DocumentReference",
                      "fhir_mapping", "to_document_reference(summary)", stage="fhir_export")

    def to_dag(self) -> dict[str, Any]:
        """Export the lineage as a serializable DAG."""
        return {
            "nodes": [
                {
                    "fqn": n.fqn,
                    "table": n.table,
                    "column": n.column,
                    "dtype": n.dtype,
                    "pii": n.pii,
                    "description": n.description,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "transformation": e.transformation,
                    "expression": e.expression,
                    "stage": e.stage,
                }
                for e in self._edges
            ],
            "stages": sorted(set(e.stage for e in self._edges if e.stage)),
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "pii_columns": len(self.get_pii_columns()),
        }

    def get_stage_lineage(self, stage: str) -> list[dict]:
        """Get all column lineages for a specific pipeline stage."""
        return [
            {
                "source": f"{cl.source_table}.{cl.source_column}",
                "target": f"{cl.target_table}.{cl.target_column}",
                "transformation": cl.transformation,
                "expression": cl.expression,
            }
            for cl in self._stage_lineages.get(stage, [])
        ]
