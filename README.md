# HERA v4 — Healthcare Reasoning & Analytics Platform

**Multi-agent clinical AI + production data engineering pipeline.**

[![CI/CD](https://github.com/Nerdboss-stm/hera-healthcare-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Nerdboss-stm/hera-healthcare-ai/actions)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-104%20passing-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Version](https://img.shields.io/badge/Version-4.0.0-orange)

**[Live Demo](https://3wihdymitc.us-east-1.awsapprunner.com)** — Running on AWS App Runner (1 vCPU, 2GB RAM)

---

## What This Is

HERA is a production-grade healthcare AI platform built in two layers:

**AI Layer** — Multi-agent clinical reasoning (Triage, Diagnostic, Treatment), RAG knowledge retrieval, biomedical NER, FHIR R4 interoperability, risk prediction, clinical summarization, and LLM-as-Judge safety evaluation.

**Data Engineering Layer** — Event streaming, column-level lineage, data quality, star schema warehouse, ETL orchestrator, CDC, and data catalog.

Both layers are unified through a **Command Center** that runs all 9 stages as a single pipeline per patient encounter.

---

## Architecture

```
                              ┌──────────────────────────────────┐
                              │        INPUT LAYER               │
                              │                                  │
                              │   Patient Vitals ──┐             │
                              │   Clinical Notes ──┼──→ FastAPI  │
                              │   FHIR R4 Bundle ──┘   (23 EP)  │
                              └──────────┬───────────────────────┘
                                         │
                              ┌──────────▼───────────────────────┐
                              │     COMMAND CENTER               │
                              │     9-Stage Unified Pipeline     │
                              │                                  │
                              │  ┌─────────────────────────────┐ │
                              │  │ 1. NER Extraction           │ │
                              │  │ 2. Knowledge Graph          │ │
                              │  │ 3. Multi-Agent Reasoning    │ │
                              │  │    ├── Triage (ESI v4)      │ │
                              │  │    ├── Diagnostic (ICD-10)  │ │
                              │  │    └── Treatment (Protocols)│ │
                              │  │ 4. Risk Prediction (RF)     │ │
                              │  │ 5. RAG Retrieval (FAISS)    │ │
                              │  │ 6. Clinical Summarization   │ │
                              │  │ 7. Safety Evaluation        │ │
                              │  │ 8. FHIR R4 Export           │ │
                              │  │ 9. Data Engineering         │ │
                              │  └─────────────────────────────┘ │
                              └──────────┬───────────────────────┘
                                         │
                    ┌────────────────────┬┴─────────────────────┐
                    │                    │                      │
         ┌──────────▼──────┐  ┌──────────▼──────┐  ┌──────────▼──────┐
         │   AI MODELS     │  │  DATA ENGINEERING│  │  INFRASTRUCTURE │
         │                 │  │                  │  │                 │
         │  Random Forest  │  │  Event Streaming │  │  PostgreSQL     │
         │  + SHAP         │  │  Column Lineage  │  │  (5 tables)     │
         │                 │  │  Data Quality    │  │                 │
         │  T5 Transformer │  │  Star Schema     │  │  Prometheus     │
         │  (Summarizer)   │  │  ETL Orchestrator│  │  Grafana        │
         │                 │  │  CDC             │  │                 │
         │  FAISS + MiniLM │  │  Data Catalog    │  │  Docker Compose │
         │  (RAG Search)   │  │                  │  │                 │
         └─────────────────┘  └──────────────────┘  └─────────────────┘
```

---

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/Nerdboss-stm/hera-healthcare-ai.git
cd hera-healthcare-ai
docker compose up --build -d
```

Services start at:
- **API + Dashboard** — http://localhost:8000
- **Swagger Docs** — http://localhost:8000/docs
- **Prometheus** — http://localhost:9090
- **Grafana** — http://localhost:3001 (admin/admin)
- **PostgreSQL** — localhost:5433 (hera/hera123)

### Seed Data

After starting services, populate the database with 20 diverse patients covering all risk levels (Critical/High/Medium/Low), ESI 1-5, various diagnoses, and CDC events:

```bash
python scripts/seed_patients.py
```

### Local

```bash
pip install -r requirements.txt
uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

---

## Deployment

### AWS App Runner (Production)

The app is deployed via GitHub Actions CI/CD to AWS App Runner:

```
Push to main → CI (lint + test + build) → Docker image → ECR → App Runner auto-deploys
```

**GitHub Secrets required:**

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user with ECR + App Runner access |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | ECR repo name (e.g. `hera-healthcare-ai`) |

**Infrastructure:**
- **App Runner**: 1 vCPU, 2GB RAM, auto-scales to 0 when idle (~$5-7/mo)
- **ECR**: Docker image registry (~$0.10/mo)
- **CI workflow**: `.github/workflows/ci.yml` — lint, test, Docker build
- **CD workflow**: `.github/workflows/deploy.yml` — build, push to ECR, trigger App Runner

---

## Try It

### Full Pipeline (Command Center)

```bash
curl -X POST http://localhost:8000/api/command-center \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "pt-001",
    "chief_complaint": "chest pain",
    "clinical_note": "65yo male with acute chest pain radiating to left arm, diaphoresis, tachycardia. History of hypertension and diabetes.",
    "heart_rate": 110,
    "respiratory_rate": 22,
    "body_temperature": 37.2,
    "oxygen_saturation": 93,
    "systolic_bp": 90,
    "diastolic_bp": 60,
    "age": 65,
    "gender": "male",
    "medical_history": ["hypertension", "diabetes"],
    "current_medications": ["aspirin", "metoprolol"],
    "allergies": ["penicillin"]
  }'
```

Returns all 9 stages: NER entities, knowledge graph, triage (ESI level), differential diagnosis (ICD-10), treatment plan, risk score, RAG citations, clinical summary, safety evaluation, FHIR bundle, and data engineering outputs.

### Individual Endpoints

```bash
# Multi-agent reasoning
curl -X POST http://localhost:8000/api/reason \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","chief_complaint":"chest pain","clinical_note":"55yo male with chest pain radiating to left arm","heart_rate":110,"systolic_bp":85,"diastolic_bp":55,"body_temperature":37.2,"respiratory_rate":24,"oxygen_saturation":92,"age":55}'

# NER extraction
curl -X POST http://localhost:8000/api/ner/extract \
  -H "Content-Type: application/json" \
  -d '{"note":"Patient on warfarin 5mg and metoprolol 50mg for atrial fibrillation."}'

# RAG knowledge query
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the treatment protocol for acute STEMI?","top_k":3}'

# Risk prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"heart_rate":110,"respiratory_rate":24,"body_temperature":37.2,"oxygen_saturation":92,"systolic_bp":85,"diastolic_bp":55,"age":60}'

# Data engineering dashboard
curl http://localhost:8000/api/de/dashboard
```

---

## API Endpoints (23)

### Clinical AI

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/command-center` | POST | Full 9-stage pipeline per patient |
| `/api/reason` | POST | Multi-agent reasoning (Triage + Dx + Tx) |
| `/api/rag/query` | POST | Semantic search over medical knowledge base |
| `/api/rag/summarize` | POST | RAG-augmented summarization with citations |
| `/api/ner/extract` | POST | Extract medications, conditions, procedures |
| `/api/ner/graph` | POST | Build patient knowledge graph |
| `/api/fhir/predict` | POST | FHIR R4 Bundle in, RiskAssessment out |
| `/api/evaluate` | POST | LLM-as-Judge safety evaluation (4-axis) |
| `/api/predict` | POST | Risk prediction from vitals |
| `/api/summarize` | POST | Clinical note summarization |
| `/api/summarize/upload` | POST | Summarize from file upload |
| `/api/health` | GET | Health check (17 services) |
| `/api/usage` | GET | Usage statistics per tenant |
| `/metrics` | GET | Prometheus metrics |

### Data Engineering

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/de/stream` | POST | Event streaming with schema validation |
| `/api/de/lineage` | GET | Column-level lineage DAG |
| `/api/de/quality` | POST | Data quality validation (12 checks) |
| `/api/de/warehouse` | GET | Star schema warehouse stats |
| `/api/de/orchestrator` | GET | ETL pipeline DAG (16 tasks) |
| `/api/de/cdc` | GET | CDC event log with replay |
| `/api/de/catalog` | GET | Data catalog (12 datasets, PII report) |
| `/api/de/dashboard` | GET | All 7 DE systems at a glance |

---

## Project Structure

```
hera-healthcare-ai/
│
├── agents/                         # Multi-Agent Clinical Reasoning
│   ├── protocols.py                #   Typed inter-agent contracts
│   ├── triage.py                   #   ESI v4 triage algorithm
│   ├── diagnostic.py               #   Differential diagnosis (ICD-10)
│   ├── treatment.py                #   Evidence-based treatment protocols
│   └── orchestrator.py             #   Pipeline coordinator + audit trail
│
├── rag/                            # RAG Medical Knowledge Base
│   ├── knowledge_base.py           #   15 curated guidelines (AHA/IDSA/ESC)
│   ├── retriever.py                #   FAISS vector search + MiniLM
│   └── rag_pipeline.py             #   Retrieve + augment + generate
│
├── ner/                            # Clinical NER + Knowledge Graph
│   ├── extractor.py                #   Entity extraction (UMLS-linked)
│   └── knowledge_graph.py          #   NetworkX patient graph
│
├── fhir_layer/                     # FHIR R4 Interoperability
│   └── converter.py                #   Bidirectional FHIR conversion
│
├── evaluation/                     # LLM-as-Judge Evaluation
│   └── evaluator.py                #   4-axis clinical safety scoring
│
├── risk_prediction/                # ML Risk Classification
│   ├── preprocessing.py            #   Feature engineering (BMI, MAP)
│   ├── train.py                    #   Random Forest training
│   ├── tune.py                     #   GridSearchCV tuning
│   └── explain.py                  #   SHAP explainability
│
├── clinical_summarizer/            # T5 Summarization
│   ├── fine_tune.py                #   T5 fine-tuning pipeline
│   ├── evaluate.py                 #   ROUGE + BERTScore eval
│   └── generate_notes.py           #   Synthetic note generator
│
├── data_engineering/               # Data Engineering (7 systems)
│   ├── streaming.py                #   Event streaming + schema registry
│   ├── lineage.py                  #   Column-level lineage DAG
│   ├── quality.py                  #   Data quality framework
│   ├── warehouse.py                #   Star schema warehouse
│   ├── orchestrator.py             #   ETL DAG (16 tasks)
│   ├── cdc.py                      #   Change data capture
│   └── catalog.py                  #   Data catalog + PII tracking
│
├── serving/                        # FastAPI Application
│   ├── api.py                      #   23 REST endpoints
│   ├── command_center.py           #   9-stage pipeline orchestrator
│   ├── schemas.py                  #   Pydantic models (30+)
│   ├── middleware.py               #   Auth, rate limiting, audit
│   ├── summarizer.py               #   T5 inference (with fallback)
│   ├── risk_predictor.py           #   Risk model inference
│   ├── db_logger.py                #   PostgreSQL logging (5 tables)
│   ├── metrics.py                  #   Prometheus metrics
│   └── static/index.html           #   Web dashboard
│
├── database/
│   └── schema.sql                  #   PostgreSQL schema (5 tables)
│
├── tests/                          # 104 tests
│   ├── test_agents.py              #   20 tests
│   ├── test_ner.py                 #   13 tests
│   ├── test_fhir.py                #   8 tests
│   ├── test_evaluation.py          #   8 tests
│   ├── test_data_engineering.py    #   41 tests
│   ├── test_api.py                 #   4 tests
│   ├── test_risk_prediction.py     #   5 tests
│   └── test_summarizer.py          #   5 tests
│
├── config/
│   ├── prometheus.yml              #   Prometheus scrape config
│   ├── alerts.yml                  #   Alert rules
│   ├── settings.py                 #   DB + path config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── prometheus.yml  #   Auto-configured Prometheus datasource
│       │   └── dashboards/
│       │       └── default.yml     #   Dashboard provisioning config
│       └── dashboards/
│           └── hera-overview.json  #   17-panel overview dashboard
│
├── scripts/
│   └── seed_patients.py            #   Seed 20 diverse patients
│
├── docker-compose.yml              #   4-service stack
├── Dockerfile                      #   API container
├── requirements.txt                #   Dependencies
├── .github/workflows/ci.yml        #   CI: lint + test + Docker
└── .github/workflows/deploy.yml    #   CD: Docker → ECR → App Runner
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Pydantic | 23 endpoints with validation |
| **Multi-Agent** | Custom Python agents | Triage, Diagnostic, Treatment |
| **RAG** | FAISS + MiniLM-L6-v2 | Semantic search over medical guidelines |
| **NER** | Regex + SciSpaCy | Biomedical entity extraction |
| **Knowledge Graph** | NetworkX | Patient relationship graph |
| **Interoperability** | HL7 FHIR R4 | Healthcare data exchange |
| **Risk Model** | scikit-learn + SHAP | Interpretable risk classification |
| **Summarization** | HuggingFace T5 | Abstractive clinical summarization |
| **Evaluation** | Custom NLI evaluator | 4-axis clinical safety scoring |
| **Streaming** | Custom Kafka-style | Schema registry + DLQ |
| **Lineage** | Custom DAG tracker | Column-level provenance |
| **Quality** | Custom GX-style | 12 checks, 5 categories |
| **Warehouse** | SQLite star schema | Fact + 4 dimension tables |
| **ETL** | Custom Airflow-style | 16-task DAG with retry + SLA |
| **CDC** | Custom capture engine | Field diffs + SHA-256 checksums |
| **Catalog** | Custom catalog service | 12 datasets, PII, freshness SLAs |
| **Database** | PostgreSQL | 5 tables for audit + logging |
| **Monitoring** | Prometheus + Grafana | Observability stack |
| **CI/CD** | GitHub Actions | Lint, test, Docker build, deploy to AWS ECR + App Runner |

---

## Database Schema (PostgreSQL)

Five tables capture all API activity:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `patient_predictions` | Risk prediction results | vitals, label, confidence |
| `summaries` | Clinical note summaries | note, summary, status |
| `clinical_reasoning_sessions` | Multi-agent audit trail | ESI, diagnosis, consensus score, JSONB audit |
| `ner_extractions` | Entity extraction results | entity count, medications, conditions (JSONB) |
| `evaluation_reports` | Safety evaluation scores | factual, hallucination, accuracy, safety |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Multi-agent over monolithic LLM | Clinical reasoning is multi-step. Agents provide separation of concerns and auditability. |
| RAG over fine-tuning for knowledge | Medical guidelines change. RAG enables hot-swap without retraining. Citations provide provenance. |
| FHIR R4 for interoperability | Mandated for US healthcare data exchange (21st Century Cures Act). |
| LLM-as-Judge over ROUGE | ROUGE can't detect hallucinations or clinically dangerous errors. |
| Random Forest + SHAP | Interpretability is non-negotiable in clinical risk scoring. |
| Unified Command Center | One POST runs all 9 stages. No manual orchestration. |
| Column-level lineage | Field-level provenance for HIPAA compliance and impact analysis. |
| Custom DE systems | Zero external dependencies. Production-faithful patterns in ~300 lines each. |

---

## Running Tests

```bash
pytest tests/ -v                              # All 104 tests
pytest tests/test_agents.py -v                # Multi-agent (20)
pytest tests/test_data_engineering.py -v      # DE systems (41)
pytest tests/test_ner.py -v                   # NER + KG (13)
pytest tests/test_fhir.py -v                  # FHIR (8)
pytest tests/test_evaluation.py -v            # Evaluator (8)

ruff check .                                  # Lint
ruff format --check .                         # Format check
```

---

## Monitoring

| Service | URL | Credentials |
|---------|-----|-------------|
| API + Dashboard | http://localhost:8000 | — |
| Swagger Docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |
| PostgreSQL | localhost:5433 | hera / hera123 |

### Prometheus

Scrapes the `/metrics` endpoint every 5s. Metrics exposed:

| Metric | Type | Description |
|--------|------|-------------|
| `hera_requests_total` | Counter | Total API requests by endpoint |
| `hera_request_failures_total` | Counter | Failed requests by endpoint |
| `hera_request_latency_seconds` | Histogram | Latency distribution (buckets: 0.1s–30s) |
| `summarizer_requests_total` | Counter | Summarizer invocations |
| `summarizer_failures_total` | Counter | Summarizer failures |
| `summarizer_request_latency_seconds` | Histogram | Summarizer latency |

**Alert rule** (`config/alerts.yml`): `TooManyHighRisk` fires when `high_risk_predictions > 3` for 30s (severity: critical).

### Grafana

Auto-provisioned datasource and dashboard via `config/grafana/provisioning/`. Dashboard JSON at `config/grafana/dashboards/hera-overview.json`.

**HERA Clinical AI - Overview** (17 panels, 4 rows):

| Row | Panels |
|-----|--------|
| **System Health** | API Uptime, Total Requests, Error Rate (5m gauge), Active Patients, P95 Latency, Total Failures |
| **Clinical Risk & Triage** | Risk Level Distribution (donut), ESI Triage Distribution (bar), Risk Predictions Over Time (stacked) |
| **API Performance** | Request Rate by Endpoint, P95 Latency by Endpoint, Error Rate by Endpoint, Latency Heatmap |
| **Pipeline & Data Engineering** | Pipeline Stage Completions (bar), CDC Events by Table, Warehouse Encounters, Request Volume (donut), Pipeline Stage Failure Rate |

---

## License

MIT
