# HERA v4 Architecture

## System Overview

HERA v4 is a unified clinical AI + data engineering platform that processes patient encounters through a 9-stage pipeline. The system combines multi-agent clinical reasoning, ML inference, RAG-grounded knowledge retrieval, and a full data engineering layer — orchestrated by a single Command Center endpoint.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                  │
│                                                                        │
│   Patient Vitals ───┐                                                  │
│   Clinical Notes ───┼──→  FastAPI (23 Endpoints)                       │
│   FHIR R4 Bundle ───┘     ├── Auth Middleware (API Key + Rate Limit)   │
│                           ├── Audit Middleware (HIPAA Trace IDs)       │
│                           └── Prometheus Metrics                       │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMMAND CENTER                                    │
│                      9-Stage Unified Pipeline                          │
│                                                                        │
│   Stage 1: NER Extraction ──────→ Regex + SciSpaCy (33 meds, 28 dx)   │
│   Stage 2: Knowledge Graph ─────→ NetworkX DiGraph (3 edge types)      │
│   Stage 3: Multi-Agent Reasoning                                       │
│            ├── Triage Agent ────→ ESI v4 (5 levels, 6 vital thresholds)│
│            ├── Diagnostic Agent → ICD-10 (6 complaint categories)      │
│            └── Treatment Agent ─→ 8 protocols + drug interactions      │
│   Stage 4: Risk Prediction ─────→ Random Forest + SHAP (9 features)    │
│   Stage 5: RAG Retrieval ───────→ FAISS + MiniLM (15 guidelines)       │
│   Stage 6: Summarization ───────→ T5 Transformer (with fallback)       │
│   Stage 7: Safety Evaluation ───→ 4-axis LLM-as-Judge                  │
│   Stage 8: FHIR R4 Export ──────→ Patient/Observation/RiskAssessment   │
│   Stage 9: Data Engineering ────→ 7 integrated DE systems              │
│                                                                        │
│   Feedback Loop: Stage 7 failure triggers Stage 6 re-summarization     │
│   with RAG context injection                                           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│   AI MODELS      │  │ DATA ENGINEERING │  │   INFRASTRUCTURE     │
│                  │  │                  │  │                      │
│  Random Forest   │  │  Event Streaming │  │  PostgreSQL (5 tbl)  │
│  + SHAP          │  │  Column Lineage  │  │  ├ predictions       │
│                  │  │  Data Quality    │  │  ├ summaries         │
│  T5 Transformer  │  │  Star Schema     │  │  ├ reasoning         │
│  (fine-tuned or  │  │  ETL Orchestrator│  │  ├ ner_extractions   │
│   t5-small)      │  │  CDC             │  │  └ evaluations       │
│                  │  │  Data Catalog    │  │                      │
│  FAISS + MiniLM  │  │                  │  │  Prometheus          │
│  (384-dim)       │  │  (all custom,    │  │  Grafana             │
│                  │  │   zero external  │  │  Docker Compose      │
│                  │  │   dependencies)  │  │                      │
└──────────────────┘  └──────────────────┘  └──────────────────────┘
```

---

## Pipeline Stages (Detail)

### Stage 1: NER Extraction

**File:** `ner/extractor.py` (277 lines)

Extracts biomedical entities from clinical notes using regex pattern matching with optional SciSpaCy augmentation.

| Category | Count | Examples |
|----------|-------|---------|
| Medications | 33 patterns | aspirin, metformin, warfarin, insulin |
| Conditions | 28 patterns | hypertension, diabetes, pneumonia |
| Procedures | 25 patterns | cardiac catheterization, intubation |
| Lab values | regex with units | troponin 2.4 ng/mL, glucose 180 mg/dL |
| Vital signs | regex with units | BP 120/80, HR 110 bpm |

All entities are mapped to UMLS-style concept categories (14 mappings). Deduplication is applied before output.

### Stage 2: Knowledge Graph

**File:** `ner/knowledge_graph.py` (247 lines)

Builds a NetworkX DiGraph from extracted entities with three relationship types:

- **Treatment relationships** (20 pairs) — e.g., aspirin → coronary artery disease
- **Contraindication relationships** (5 pairs) — e.g., aspirin → bleeding disorder
- **Diagnostic relationships** (9 pairs) — e.g., troponin → myocardial infarction

Supports `query_entity()`, `get_interactions()`, `get_treatment_chain()`, and `to_cytoscape()` export.

### Stage 3: Multi-Agent Reasoning

**Files:** `agents/triage.py` (265), `agents/diagnostic.py` (334), `agents/treatment.py` (467), `agents/orchestrator.py` (113)

Three agents process sequentially with typed protocol contracts:

**Triage Agent (ESI v4)**
- 6 vital sign thresholds × 4 levels each
- High-acuity complaint keywords (ESI-1: cardiac arrest, unresponsive; ESI-2: chest pain, stroke symptoms)
- Resource estimation for ESI-3/4/5 differentiation
- Risk score from ML predictor with heuristic fallback

**Diagnostic Agent (ICD-10)**
- 6 complaint categories mapped to differential diagnoses
- Probability scoring: evidence overlap − rule-out penalty + acuity boost + age factor
- Recommended tests keyed by ICD-10 code

**Treatment Agent**
- 8 treatment protocols by ICD-10 (ACS, PE, Pneumonia, COPD, HF, Sepsis, Stroke, Appendicitis)
- Drug interaction checking (5 known interaction pairs)
- Allergy cross-reference and contraindication filtering
- Geriatric/pediatric dose precautions

**Orchestrator** chains all three and records audit trail. Consensus score = 0.3×risk + 0.4×confidence + 0.3×evidence_grade.

### Stage 4: Risk Prediction

**File:** `serving/risk_predictor.py` (93 lines)

Random Forest classifier with 9 features:
- 7 raw vitals (HR, RR, temp, SpO2, SBP, DBP, age)
- Calculated BMI (weight/height²)
- Calculated MAP (DBP + (SBP−DBP)/3)

SHAP TreeExplainer provides exact Shapley values. Output: binary risk label + confidence + feature importances.

### Stage 5: RAG Retrieval

**Files:** `rag/knowledge_base.py` (194), `rag/retriever.py` (133), `rag/rag_pipeline.py` (118)

- **Knowledge base:** 15 curated entries across cardiology (3), pulmonology (3), emergency medicine (3), neurology (2), pharmacology (3), infectious disease (1), surgery (1)
- **Retriever:** FAISS IndexFlatIP with all-MiniLM-L6-v2 encoder (384-dim). L2 normalization for cosine similarity.
- **Pipeline:** Augments query with "[source]: text" context blocks before generation

### Stage 6: Clinical Summarization

**File:** `serving/summarizer.py` (68 lines)

T5 encoder-decoder with automatic fallback:
1. Attempts to load fine-tuned model from `model/` directory
2. Validates output (checks for non-empty text)
3. Falls back to pretrained `t5-small` if fine-tuned model produces empty output
4. Uses `summarize:` task prefix for pretrained model
5. Beam search (num_beams=4) with length penalty and no-repeat n-gram

### Stage 7: Safety Evaluation

**File:** `evaluation/evaluator.py` (451 lines)

LLM-as-Judge scores outputs on 4 axes:

| Axis | Weight | Method |
|------|--------|--------|
| Factual consistency | 0.30 | Keyword overlap + 7 contradiction pair detection |
| Hallucination | 0.25 | Entity precision/recall + fabricated claim detection |
| Medical accuracy | 0.25 | 157 valid terms + dangerous dosage checks (5 drugs) |
| Clinical safety | 0.20 | 3 dangerous regex patterns + allergy cross-check |

Overall score = weighted sum. If score < threshold, triggers re-summarization with RAG context (feedback loop to Stage 6).

### Stage 8: FHIR R4 Export

**File:** `fhir_layer/converter.py` (296 lines)

Bidirectional conversion with LOINC-coded observations:

| LOINC Code | Vital Sign |
|------------|-----------|
| 8867-4 | Heart rate |
| 8480-6 | Systolic BP |
| 8462-4 | Diastolic BP |
| 8310-5 | Body temperature |
| 9279-1 | Respiratory rate |
| 2708-6 | SpO2 |
| 29463-7 | Body weight |
| 8302-2 | Body height |

Output FHIR resources: Patient, Observation (per vital), RiskAssessment, DocumentReference (LOINC 34133-9).

### Stage 9: Data Engineering

Seven integrated systems process pipeline output:

---

## Data Engineering Layer

### 1. Event Streaming (`streaming.py`, 422 lines)

Kafka-style event streaming with:
- **Schema Registry** — 5 default schemas (patient_vitals v1/v2/v3, clinical_note v1, pipeline_result v1)
- **MD5 partitioning** — Deterministic partition assignment
- **Consumer groups** — Named consumers with offset tracking
- **Dead Letter Queue** — Captures schema validation failures
- **Event processing** — Computes MAP and BMI on ingestion

### 2. Column-Level Lineage (`lineage.py`, 506 lines)

DAG tracking field transformations across all 8 pipeline stages:
- **36+ nodes** (fully qualified: stage.column)
- **38+ edges** (with transformation labels)
- `get_upstream()` / `get_downstream()` — recursive tracing
- `impact_analysis()` — what breaks if a source column changes
- **PII flagging** on patient_id, clinical_note, medications, summary_text

### 3. Data Quality (`quality.py`, 349 lines)

Great Expectations-style framework:
- **12 checks** across 5 categories (schema, completeness, accuracy, consistency, freshness)
- Clinical vital sign range validation (7 vitals)
- BP consistency check (systolic > diastolic)
- Pulse pressure validation
- SSN pattern detection in clinical notes
- Weighted scoring: info=0.5, warning=1.0, critical=2.0

### 4. Star Schema Warehouse (`warehouse.py`, 394 lines)

SQLite star schema:

```
fact_clinical_encounters (22 columns)
├── dim_patient       (patient_id, age, gender)
├── dim_diagnosis     (icd10_code, description, category)
├── dim_provider      (9 HERA agents pre-seeded)
└── dim_time          (year, quarter, month, day, hour, day_of_week)

agg_hourly_summary    (encounter_count, avg_risk, avg_quality, avg_processing_ms)
```

3 indexes. Hourly aggregation via SQL refresh.

### 5. ETL Orchestrator (`orchestrator.py`, 437 lines)

Airflow-style DAG execution:
- **16 tasks** with dependency graph
- Topological sort for execution order
- Configurable retry with delay per task
- SLA monitoring
- Upstream failure propagation (if triage fails, diagnosis is skipped)

```
ingest → validate → extract_entities → build_kg
                  → run_triage → run_diagnosis → run_treatment
                  → predict_risk
                                  run_diagnosis → retrieve_rag → generate_summary → evaluate_safety
                                                                                            ↓
export_fhir ← [run_treatment, predict_risk, generate_summary, evaluate_safety]
    ├── load_warehouse → emit_cdc → update_catalog
    └── track_lineage → update_catalog
```

### 6. Change Data Capture (`cdc.py`, 246 lines)

- Before/after snapshots with field-level diffs
- SHA-256 checksums for integrity
- Named consumers with offset tracking
- Event replay from any sequence number
- Record history queries
- Per-table and per-type statistics

### 7. Data Catalog (`catalog.py`, 499 lines)

12 pre-registered datasets:

| Dataset | PII | SLA |
|---------|-----|-----|
| raw_patient_vitals | Yes | 1 hour |
| clinical_notes | Yes | 1 hour |
| ner_entities | No | 4 hours |
| knowledge_graph | No | 4 hours |
| agent_reasoning | No | 4 hours |
| risk_predictions | No | 1 hour |
| rag_results | No | 4 hours |
| clinical_summaries | Yes | 4 hours |
| safety_evaluations | No | 4 hours |
| fhir_bundles | Yes | 1 hour |
| fact_encounters | No | 1 hour |
| cdc_event_log | Yes | 1 hour |

Searchable with relevance scoring. PII report and freshness report APIs.

---

## PostgreSQL Schema

5 tables capture all API activity:

```sql
patient_predictions       -- Risk prediction results (12 columns)
summaries                 -- Clinical note summaries (4 columns)
clinical_reasoning_sessions -- Multi-agent audit trail (10 columns, JSONB audit)
ner_extractions           -- Entity extraction results (8 columns, JSONB entities)
evaluation_reports        -- Safety evaluation scores (9 columns, JSONB details)
```

All 5 tables are populated automatically when API endpoints are called. DB logging is best-effort (non-blocking).

---

## Middleware Stack

| Middleware | Purpose |
|-----------|---------|
| **AuthMiddleware** | API key authentication, tenant isolation, rate limiting (30/min anonymous, 60/min authenticated) |
| **AuditMiddleware** | HIPAA-compliant audit logging with trace IDs, latency tracking |
| **UsageTracker** | Per-tenant usage metering (requests/minute, daily, endpoint breakdown) |

Bypass paths: `/`, `/api/health`, `/docs`, `/metrics`, `/static/*`, `/api/de/*`

---

## Design Tradeoffs

### Unified Command Center over Microservices
One `POST /api/command-center` runs all 9 stages per patient. Simplifies deployment and ensures atomic processing. Decomposition into microservices at 100K+ patients/day.

### Custom DE Systems over Airflow/Kafka/dbt
Each system is ~200-400 lines of focused Python. Zero external dependencies. Production-faithful patterns (schema registry, DAG execution, CDC capture) that run with just `pip install`.

### SQLite Warehouse over PostgreSQL
Identical star schema design. Switching backends requires only a connection string change. Zero-dependency local operation for development and testing.

### T5-small with Fallback
Fine-tuned model is loaded first. If it produces empty output (training didn't converge), automatic fallback to pretrained t5-small with `summarize:` task prefix. No silent failures.

### Random Forest + SHAP over Deep Learning
9 features, binary target. SHAP TreeExplainer gives exact Shapley values. "SpO2 and MAP drove this High Risk" is more valuable than marginal accuracy from a black-box model.

### Column-Level over Table-Level Lineage
Field-level provenance enables precise impact analysis and HIPAA-compliant PII tracking. Table-level is too coarse for healthcare governance.

---

## Scalability Path

| Scale | Changes |
|-------|---------|
| **Current** (single server) | 23 endpoints, 104 tests, all in-process |
| **10K/day** | Connection pooling (pgBouncer), async DB logging, multi-worker uvicorn |
| **100K/day** | SQLite → PostgreSQL/Redshift, GPU inference for T5, horizontal FastAPI behind LB |
| **1M+/day** | Microservices, Kafka, Airflow, Triton/TorchServe, BigQuery/Snowflake |

---

## Service Health (17 Systems)

`GET /api/health` reports on:

1. command_center
2. summarizer
3. risk_predictor
4. clinical_reasoning
5. rag_knowledge_base
6. ner_extraction
7. fhir_converter
8. clinical_evaluator
9. auth_middleware
10. usage_metering
11. event_streaming
12. data_lineage
13. data_quality
14. analytics_warehouse
15. etl_orchestrator
16. cdc_stream
17. data_catalog
