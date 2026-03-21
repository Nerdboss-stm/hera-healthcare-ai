"""Data catalog & metadata management — self-documenting data platform.

Demonstrates: automatic schema discovery, field-level documentation,
PII classification, freshness monitoring, searchable catalog API,
and data-as-a-product thinking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ColumnMetadata:
    """Metadata for a single column/field."""

    name: str
    dtype: str
    description: str = ""
    pii: bool = False
    nullable: bool = True
    example: str = ""
    source: str = ""  # which system produces this
    tags: list[str] = field(default_factory=list)


@dataclass
class DatasetEntry:
    """A registered dataset in the catalog."""

    dataset_id: str
    name: str
    description: str
    owner: str  # system/team that owns this
    category: str  # raw, derived, aggregated, exported
    columns: list[ColumnMetadata] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    freshness_sla_minutes: int = 60
    last_updated: str = ""
    row_count: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.last_updated:
            self.last_updated = self.created_at


class DataCatalog:
    """Searchable data catalog for the HERA platform."""

    def __init__(self):
        self._datasets: dict[str, DatasetEntry] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all HERA pipeline datasets."""
        self.register(DatasetEntry(
            dataset_id="raw_patient_vitals",
            name="Raw Patient Vitals",
            description="Real-time patient vital signs ingested through the event streaming pipeline",
            owner="streaming_pipeline",
            category="raw",
            tags=["vitals", "real-time", "streaming", "patient-data"],
            columns=[
                ColumnMetadata("patient_id", "str", "Unique patient identifier", pii=True, nullable=False, example="P-001"),
                ColumnMetadata("heart_rate", "float", "Heart rate in BPM", example="88.0"),
                ColumnMetadata("respiratory_rate", "float", "Breaths per minute", example="18.0"),
                ColumnMetadata("body_temperature", "float", "Temperature in Celsius", example="37.2"),
                ColumnMetadata("oxygen_saturation", "float", "SpO2 percentage", example="96.0"),
                ColumnMetadata("systolic_bp", "float", "Systolic blood pressure mmHg", example="142.0"),
                ColumnMetadata("diastolic_bp", "float", "Diastolic blood pressure mmHg", example="88.0"),
                ColumnMetadata("age", "int", "Patient age in years", example="65"),
                ColumnMetadata("chief_complaint", "str", "Primary complaint text", example="chest pain"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="clinical_notes",
            name="Clinical Notes",
            description="Free-text clinical notes from patient encounters",
            owner="ehr_system",
            category="raw",
            tags=["notes", "text", "phi", "clinical"],
            columns=[
                ColumnMetadata("patient_id", "str", "Patient identifier", pii=True, nullable=False),
                ColumnMetadata("note_text", "str", "Full clinical note text", pii=True),
                ColumnMetadata("encounter_id", "str", "Encounter reference"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="ner_entities",
            name="NER Extracted Entities",
            description="Medical entities extracted from clinical notes via regex-based NER",
            owner="ner_extractor",
            category="derived",
            tags=["ner", "entities", "medications", "conditions", "derived"],
            columns=[
                ColumnMetadata("entity_count", "int", "Total entities found"),
                ColumnMetadata("medications", "list[dict]", "Extracted medication names", pii=True),
                ColumnMetadata("conditions", "list[dict]", "Diagnosed conditions"),
                ColumnMetadata("procedures", "list[dict]", "Medical procedures mentioned"),
                ColumnMetadata("lab_values", "list[dict]", "Lab test results"),
                ColumnMetadata("vital_signs", "list[dict]", "Vital sign mentions in text"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="knowledge_graph",
            name="Patient Knowledge Graph",
            description="NetworkX graph of entity relationships per patient",
            owner="ner_extractor",
            category="derived",
            tags=["graph", "networkx", "relationships", "derived"],
            columns=[
                ColumnMetadata("nodes", "int", "Graph node count"),
                ColumnMetadata("edges", "int", "Graph edge count"),
                ColumnMetadata("graph_data", "dict", "Serialized graph structure"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="agent_reasoning",
            name="Multi-Agent Reasoning Results",
            description="Triage + Diagnostic + Treatment agent outputs with consensus scoring",
            owner="clinical_orchestrator",
            category="derived",
            tags=["agents", "triage", "diagnosis", "treatment", "reasoning"],
            columns=[
                ColumnMetadata("esi_level", "int", "Emergency Severity Index (1-5)"),
                ColumnMetadata("primary_diagnosis", "str", "Primary diagnosis from diagnostic agent"),
                ColumnMetadata("disposition", "str", "Recommended care disposition"),
                ColumnMetadata("consensus_score", "float", "Agent agreement score [0, 1]"),
                ColumnMetadata("reasoning_audit", "list[dict]", "Full reasoning chain"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="risk_predictions",
            name="ML Risk Predictions",
            description="Random Forest risk scores with derived features (BMI, MAP)",
            owner="risk_predictor",
            category="derived",
            tags=["ml", "risk", "prediction", "random-forest"],
            columns=[
                ColumnMetadata("risk_score", "float", "Probability of adverse outcome [0, 1]"),
                ColumnMetadata("prediction", "str", "Binary classification: High Risk / Low Risk"),
                ColumnMetadata("confidence", "float", "Model confidence [0, 1]"),
                ColumnMetadata("features_used", "dict", "Input features including derived"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="rag_results",
            name="RAG Retrieved Context",
            description="Medical guidelines retrieved via FAISS semantic vector search",
            owner="rag_pipeline",
            category="derived",
            tags=["rag", "vector-search", "faiss", "guidelines"],
            columns=[
                ColumnMetadata("results", "list[dict]", "Top-K retrieved passages"),
                ColumnMetadata("total_corpus_size", "int", "Total docs in knowledge base"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="clinical_summaries",
            name="Clinical Summaries",
            description="T5-generated summaries of clinical notes with compression metrics",
            owner="summarizer",
            category="derived",
            tags=["summary", "t5", "nlp", "compression"],
            columns=[
                ColumnMetadata("summary", "str", "Generated summary text", pii=True),
                ColumnMetadata("compression", "float", "Compression ratio percentage"),
                ColumnMetadata("input_length", "int", "Source note character count"),
                ColumnMetadata("summary_length", "int", "Summary character count"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="safety_evaluations",
            name="Safety Evaluation Reports",
            description="LLM-as-Judge 4-axis evaluation of clinical AI outputs",
            owner="clinical_evaluator",
            category="derived",
            tags=["safety", "evaluation", "hallucination", "quality"],
            columns=[
                ColumnMetadata("overall_score", "float", "Weighted evaluation score [0, 1]"),
                ColumnMetadata("factual_consistency", "dict", "Source-summary alignment score"),
                ColumnMetadata("hallucination", "dict", "Hallucination detection result"),
                ColumnMetadata("medical_accuracy", "dict", "Clinical correctness score"),
                ColumnMetadata("clinical_safety", "dict", "Patient safety assessment"),
                ColumnMetadata("pass_threshold", "bool", "Whether output passes quality gate"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="fhir_bundles",
            name="FHIR R4 Bundles",
            description="HL7 FHIR R4 compliant bundles for EHR interoperability",
            owner="fhir_converter",
            category="exported",
            tags=["fhir", "hl7", "interoperability", "ehr", "export"],
            columns=[
                ColumnMetadata("bundle", "dict", "Complete FHIR R4 Bundle resource"),
                ColumnMetadata("resource_count", "int", "Number of FHIR resources"),
                ColumnMetadata("resource_types", "list[str]", "Types of resources included"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="fact_encounters",
            name="Fact: Clinical Encounters",
            description="Star schema fact table with all encounter metrics for analytics",
            owner="analytics_warehouse",
            category="aggregated",
            tags=["warehouse", "star-schema", "fact-table", "analytics"],
            columns=[
                ColumnMetadata("encounter_id", "int", "Auto-incremented encounter key"),
                ColumnMetadata("patient_key", "int", "FK to dim_patient"),
                ColumnMetadata("diagnosis_key", "int", "FK to dim_diagnosis"),
                ColumnMetadata("risk_score", "float", "ML risk prediction score"),
                ColumnMetadata("safety_score", "float", "Safety evaluation score"),
                ColumnMetadata("pipeline_latency_ms", "float", "End-to-end pipeline latency"),
                ColumnMetadata("fhir_resource_count", "int", "FHIR resources generated"),
            ],
        ))

        self.register(DatasetEntry(
            dataset_id="cdc_event_log",
            name="CDC Event Log",
            description="Immutable change data capture log for all state changes",
            owner="cdc_stream",
            category="raw",
            tags=["cdc", "audit", "event-sourcing", "immutable"],
            columns=[
                ColumnMetadata("event_id", "str", "Unique CDC event identifier"),
                ColumnMetadata("table", "str", "Source table name"),
                ColumnMetadata("change_type", "str", "INSERT, UPDATE, or DELETE"),
                ColumnMetadata("before", "dict", "State before change"),
                ColumnMetadata("after", "dict", "State after change"),
                ColumnMetadata("diff", "dict", "Field-level differences"),
                ColumnMetadata("checksum", "str", "SHA-256 integrity checksum"),
            ],
        ))

    def register(self, dataset: DatasetEntry):
        self._datasets[dataset.dataset_id] = dataset

    def get(self, dataset_id: str) -> DatasetEntry | None:
        return self._datasets.get(dataset_id)

    def search(self, query: str) -> list[dict]:
        """Search datasets by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for ds in self._datasets.values():
            score = 0
            if query_lower in ds.name.lower():
                score += 3
            if query_lower in ds.description.lower():
                score += 2
            if any(query_lower in tag for tag in ds.tags):
                score += 1
            if any(query_lower in col.name for col in ds.columns):
                score += 1
            if score > 0:
                results.append({
                    "dataset_id": ds.dataset_id,
                    "name": ds.name,
                    "description": ds.description,
                    "category": ds.category,
                    "tags": ds.tags,
                    "relevance_score": score,
                })
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)

    def get_pii_report(self) -> list[dict]:
        """List all PII fields across the platform."""
        pii_fields = []
        for ds in self._datasets.values():
            for col in ds.columns:
                if col.pii:
                    pii_fields.append({
                        "dataset": ds.dataset_id,
                        "column": col.name,
                        "dtype": col.dtype,
                        "description": col.description,
                        "owner": ds.owner,
                    })
        return pii_fields

    def get_freshness_report(self) -> list[dict]:
        """Check which datasets are stale."""
        now = datetime.now(timezone.utc)
        report = []
        for ds in self._datasets.values():
            if ds.last_updated:
                try:
                    last = datetime.fromisoformat(ds.last_updated.replace("Z", "+00:00"))
                    age_minutes = (now - last).total_seconds() / 60
                    is_stale = age_minutes > ds.freshness_sla_minutes
                except (ValueError, TypeError):
                    age_minutes = -1
                    is_stale = True
            else:
                age_minutes = -1
                is_stale = True

            report.append({
                "dataset_id": ds.dataset_id,
                "name": ds.name,
                "sla_minutes": ds.freshness_sla_minutes,
                "age_minutes": round(age_minutes, 1),
                "is_stale": is_stale,
            })
        return report

    def update_freshness(self, dataset_id: str, row_count: int = 0):
        """Mark a dataset as freshly updated."""
        ds = self._datasets.get(dataset_id)
        if ds:
            ds.last_updated = datetime.now(timezone.utc).isoformat()
            if row_count:
                ds.row_count = row_count

    def to_dict(self) -> dict:
        """Export full catalog."""
        return {
            "datasets": {
                ds.dataset_id: {
                    "name": ds.name,
                    "description": ds.description,
                    "owner": ds.owner,
                    "category": ds.category,
                    "tags": ds.tags,
                    "columns": [
                        {
                            "name": c.name,
                            "dtype": c.dtype,
                            "description": c.description,
                            "pii": c.pii,
                        }
                        for c in ds.columns
                    ],
                    "row_count": ds.row_count,
                    "last_updated": ds.last_updated,
                }
                for ds in self._datasets.values()
            },
            "total_datasets": len(self._datasets),
            "total_columns": sum(len(ds.columns) for ds in self._datasets.values()),
            "pii_columns": len(self.get_pii_report()),
            "categories": sorted(set(ds.category for ds in self._datasets.values())),
        }


# Singleton
data_catalog = DataCatalog()
