import os
import time
import hashlib
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse, FileResponse
from serving.schemas import (
    NoteRequest,
    SummaryResponse,
    VitalsRequest,
    RiskResponse,
    HealthResponse,
    ClinicalReasoningRequest,
    ClinicalReasoningResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGSummarizeRequest,
    RAGSummarizeResponse,
    NERRequest,
    NERResponse,
    KnowledgeGraphRequest,
    KnowledgeGraphResponse,
    FHIRBundleRequest,
    FHIRPredictResponse,
    EvaluationRequest,
    EvaluationResponse,
    CommandCenterRequest,
    CommandCenterResponse,
    UsageResponse,
    DEStreamingResponse,
    DELineageResponse,
    DEQualityResponse,
    DEWarehouseResponse,
    DEOrchestratorResponse,
    DECDCResponse,
    DECatalogResponse,
    DEDashboardResponse,
)
from serving.summarizer import generate_summary
from serving.risk_predictor import predict_risk
from serving.db_logger import (
    log_to_db,
    log_prediction,
    log_reasoning,
    log_ner,
    log_evaluation,
)
from serving.metrics import (
    REQUEST_COUNT,
    REQUEST_FAILURES,
    REQUEST_LATENCY,
    prometheus_metrics,
)

_dir = os.path.dirname(os.path.abspath(__file__))

from serving.middleware import AuthMiddleware, AuditMiddleware, usage_tracker  # noqa: E402

app = FastAPI(
    title="HERA — Healthcare Risk Analytics",
    description=(
        "Production-grade multi-agent clinical AI platform with unified pipeline, "
        "RAG knowledge retrieval, NER entity extraction, FHIR R4 interoperability, "
        "LLM-as-Judge evaluation, API key auth, rate limiting, and usage metering."
    ),
    version="4.0.0",
)

app.add_middleware(AuditMiddleware)
app.add_middleware(AuthMiddleware)

# Serve the frontend
app.mount("/static", StaticFiles(directory=os.path.join(_dir, "static")), name="static")


# ── Health & Metrics ──────────────────────────────────────────────


@app.get("/", response_class=FileResponse)
def serve_ui():
    return FileResponse(
        os.path.join(_dir, "static", "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    summarizer_status = "available"
    try:
        from serving.summarizer import _load_model

        _load_model()
    except Exception:
        summarizer_status = "model not loaded"

    risk_status = "available"
    try:
        from serving.risk_predictor import _load_risk_model

        _load_risk_model()
    except Exception:
        risk_status = "model not loaded"

    return HealthResponse(
        status="healthy",
        services={
            "command_center": "available",
            "summarizer": summarizer_status,
            "risk_predictor": risk_status,
            "clinical_reasoning": "available",
            "rag_knowledge_base": "available",
            "ner_extraction": "available",
            "fhir_converter": "available",
            "clinical_evaluator": "available",
            "auth_middleware": "available",
            "usage_metering": "available",
            "event_streaming": "available",
            "data_lineage": "available",
            "data_quality": "available",
            "analytics_warehouse": "available",
            "etl_orchestrator": "available",
            "cdc_stream": "available",
            "data_catalog": "available",
        },
        version="4.0.0",
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return prometheus_metrics()


# ── Clinical Summarization ────────────────────────────────────────


@app.post("/api/summarize", response_model=SummaryResponse)
def summarize_note(request: NoteRequest):
    ep = "summarize"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        summary = generate_summary(request.note)
        log_to_db(request.note, summary, "SUCCESS")
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return SummaryResponse(
            summary=summary,
            timestamp=datetime.now(),
            note_length=len(request.note),
            summary_length=len(summary),
        )
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        log_to_db(request.note, str(e), "FAILURE")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize/upload", response_model=SummaryResponse)
async def summarize_file(file: UploadFile = File(...)):
    ep = "summarize_upload"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        content = await file.read()
        note_text = content.decode("utf-8").strip()
        if len(note_text) < 10:
            raise HTTPException(status_code=400, detail="File content too short")
        summary = generate_summary(note_text)
        log_to_db(note_text, summary, "SUCCESS")
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return SummaryResponse(
            summary=summary,
            timestamp=datetime.now(),
            note_length=len(note_text),
            summary_length=len(summary),
        )
    except UnicodeDecodeError:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        raise HTTPException(status_code=400, detail="File must be a UTF-8 text file")
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── Risk Prediction ──────────────────────────────────────────────


@app.post("/api/predict", response_model=RiskResponse)
def predict_patient_risk(request: VitalsRequest):
    ep = "predict"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        result = predict_risk(
            heart_rate=request.heart_rate,
            respiratory_rate=request.respiratory_rate,
            body_temperature=request.body_temperature,
            oxygen_saturation=request.oxygen_saturation,
            systolic_bp=request.systolic_bp,
            diastolic_bp=request.diastolic_bp,
            age=request.age,
        )
        log_prediction(result["features_used"], result["prediction"], result["confidence"])
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return RiskResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            features_used=result["features_used"],
            timestamp=datetime.now(),
        )
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── Multi-Agent Clinical Reasoning ───────────────────────────────


@app.post("/api/reason", response_model=ClinicalReasoningResponse)
def clinical_reasoning(request: ClinicalReasoningRequest):
    """Run the multi-agent clinical reasoning pipeline.

    Triage Agent → Diagnostic Agent → Treatment Agent, producing
    a full clinical assessment with auditable reasoning chain.
    """
    ep = "reason"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        from agents.orchestrator import ClinicalOrchestrator
        from agents.protocols import PatientContext

        ctx = PatientContext(
            patient_id=request.patient_id,
            chief_complaint=request.chief_complaint,
            clinical_note=request.clinical_note,
            vitals={
                "heart_rate": request.heart_rate,
                "respiratory_rate": request.respiratory_rate,
                "body_temperature": request.body_temperature,
                "oxygen_saturation": request.oxygen_saturation,
                "systolic_bp": request.systolic_bp,
                "diastolic_bp": request.diastolic_bp,
            },
            age=request.age,
            gender=request.gender,
            medical_history=request.medical_history,
            current_medications=request.current_medications,
            allergies=request.allergies,
        )

        orchestrator = ClinicalOrchestrator(risk_predictor=predict_risk)
        result = orchestrator.reason(ctx)

        triage_dict = result.triage.to_dict()
        diag_dict = result.diagnosis.to_dict()
        treat_dict = result.treatment.to_dict()

        # Extract diagnosis — may be a string or dict depending on to_dict()
        primary_dx = diag_dict.get("primary_diagnosis", "")
        dx_name = primary_dx.get("name", "") if isinstance(primary_dx, dict) else str(primary_dx)
        dx_conf = primary_dx.get("probability", 0) if isinstance(primary_dx, dict) else diag_dict.get("confidence", 0)

        try:
            log_reasoning(
                patient_id=result.patient_id,
                complaint=request.chief_complaint,
                esi=triage_dict.get("esi_level", 0),
                diagnosis=dx_name,
                confidence=dx_conf,
                disposition=treat_dict.get("disposition", ""),
                consensus=result.consensus_score,
                latency=result.pipeline_latency_ms,
                audit=result.reasoning_audit,
            )
        except Exception:
            pass  # DB logging is best-effort

        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return ClinicalReasoningResponse(
            patient_id=result.patient_id,
            triage=triage_dict,
            diagnosis=diag_dict,
            treatment=treat_dict,
            reasoning_audit=result.reasoning_audit,
            consensus_score=result.consensus_score,
            pipeline_latency_ms=result.pipeline_latency_ms,
            timestamp=datetime.now(),
        )
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── RAG Medical Knowledge Base ───────────────────────────────────

_rag_pipeline = None


def _get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            from rag.rag_pipeline import RAGPipeline

            _rag_pipeline = RAGPipeline(summarizer_fn=generate_summary)
            _rag_pipeline.initialize()
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="RAG dependencies not installed (faiss-cpu, sentence-transformers)",
            )
    return _rag_pipeline


@app.post("/api/rag/query", response_model=RAGQueryResponse)
def rag_query(request: RAGQueryRequest):
    """Query the medical knowledge base using semantic search."""
    ep = "rag_query"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        rag = _get_rag_pipeline()
        result = rag.query_knowledge(request.query, top_k=request.top_k)
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return RAGQueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/summarize", response_model=RAGSummarizeResponse)
def rag_summarize(request: RAGSummarizeRequest):
    """RAG-augmented clinical note summarization with citations."""
    ep = "rag_summarize"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        rag = _get_rag_pipeline()
        result = rag.augment_and_generate(request.note, top_k=request.top_k)
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return RAGSummarizeResponse(
            **result,
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── Clinical NER & Knowledge Graph ───────────────────────────────


@app.post("/api/ner/extract", response_model=NERResponse)
def extract_entities(request: NERRequest):
    """Extract medical entities (medications, conditions, procedures, labs) from clinical notes."""
    ep = "ner_extract"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        from ner.extractor import ClinicalNERExtractor

        extractor = ClinicalNERExtractor()
        result = extractor.extract(request.note)
        result_dict = result.to_dict()

        note_hash = hashlib.sha256(request.note.encode()).hexdigest()[:16]
        try:
            log_ner(
                patient_id=getattr(request, "patient_id", "unknown"),
                note_hash=note_hash,
                count=result_dict.get("entity_count", 0),
                medications=result_dict.get("medications", []),
                conditions=result_dict.get("conditions", []),
                procedures=result_dict.get("procedures", []),
                labs=result_dict.get("lab_values", []),
            )
        except Exception:
            pass  # DB logging is best-effort

        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return NERResponse(
            **result_dict,
            timestamp=datetime.now(),
        )
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ner/graph", response_model=KnowledgeGraphResponse)
def build_knowledge_graph(request: KnowledgeGraphRequest):
    """Build a patient knowledge graph from a clinical note."""
    ep = "ner_graph"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        from ner.knowledge_graph import PatientKnowledgeGraph

        kg = PatientKnowledgeGraph()
        result = kg.build_from_note(request.note, request.patient_id)
        graph_data = kg.to_dict()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return KnowledgeGraphResponse(
            patient_id=result["patient_id"],
            nodes=result["nodes"],
            edges=result["edges"],
            entities=result["entities"],
            graph=graph_data,
            timestamp=datetime.now(),
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="networkx not installed")
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── FHIR R4 Interoperability ────────────────────────────────────


@app.post("/api/fhir/predict", response_model=FHIRPredictResponse)
def fhir_predict(request: FHIRBundleRequest):
    """Accept a FHIR R4 Bundle, run risk prediction, return FHIR RiskAssessment."""
    ep = "fhir_predict"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        from fhir_layer.converter import FHIRConverter

        parsed = FHIRConverter.parse_bundle(request.bundle)
        patient = parsed["patient"]
        vitals = parsed["vitals"]

        result = predict_risk(
            heart_rate=vitals.get("heart_rate", 80),
            respiratory_rate=vitals.get("respiratory_rate", 16),
            body_temperature=vitals.get("body_temperature", 37.0),
            oxygen_saturation=vitals.get("oxygen_saturation", 98),
            systolic_bp=vitals.get("systolic_bp", 120),
            diastolic_bp=vitals.get("diastolic_bp", 80),
            age=patient.get("age", 50),
        )

        log_prediction(result["features_used"], result["prediction"], result["confidence"])

        risk_assessment = FHIRConverter.to_risk_assessment(
            patient_id=patient.get("patient_id", "unknown"),
            prediction=result["prediction"],
            risk_score=result["risk_score"],
            confidence=result["confidence"],
            features=result["features_used"],
        )

        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return FHIRPredictResponse(
            risk_assessment=risk_assessment,
            original_prediction=result,
            timestamp=datetime.now(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── LLM-as-Judge Evaluation ─────────────────────────────────────


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate_output(request: EvaluationRequest):
    """Evaluate a clinical AI output for factual consistency, hallucinations, and safety."""
    ep = "evaluate"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        from evaluation.evaluator import ClinicalEvaluator

        evaluator = ClinicalEvaluator()
        report = evaluator.evaluate(request.source_note, request.generated_output)
        report_dict = report.to_dict()

        note_hash = hashlib.sha256(request.source_note.encode()).hexdigest()[:16]
        try:
            log_evaluation(
                note_hash=note_hash,
                overall=report_dict.get("overall_score", 0),
                passed=report_dict.get("pass_threshold", False),
                factual=report_dict.get("factual_consistency", {}).get("score", 0),
                hallucination=report_dict.get("hallucination", {}).get("score", 0),
                accuracy=report_dict.get("medical_accuracy", {}).get("score", 0),
                safety=report_dict.get("clinical_safety", {}).get("safe", False),
                details=report_dict,
            )
        except Exception:
            pass  # DB logging is best-effort

        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return EvaluationResponse(
            **report_dict,
            timestamp=datetime.now(),
        )
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── Unified Command Center ─────────────────────────────────────


@app.post("/api/command-center", response_model=CommandCenterResponse)
def command_center(request: CommandCenterRequest):
    """Run the unified pipeline — all 9 stages on one patient.

    NER → Knowledge Graph → Multi-Agent Reasoning → Risk Prediction →
    RAG Retrieval → Summarization → Safety Evaluation → FHIR Export →
    Data Engineering.
    """
    ep = "command_center"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        from serving.command_center import UnifiedPipeline

        pipeline = UnifiedPipeline(
            risk_predictor=predict_risk,
            summarizer_fn=generate_summary,
        )
        result = pipeline.execute(
            patient_id=request.patient_id,
            clinical_note=request.clinical_note,
            chief_complaint=request.chief_complaint,
            vitals={
                "heart_rate": request.heart_rate,
                "respiratory_rate": request.respiratory_rate,
                "body_temperature": request.body_temperature,
                "oxygen_saturation": request.oxygen_saturation,
                "systolic_bp": request.systolic_bp,
                "diastolic_bp": request.diastolic_bp,
            },
            age=request.age,
            gender=request.gender,
            medical_history=request.medical_history,
            current_medications=request.current_medications,
            allergies=request.allergies,
        )
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return CommandCenterResponse(
            **result.to_dict(),
            timestamp=datetime.now(),
        )
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))


# ── Usage & Admin ──────────────────────────────────────────────


@app.get("/api/usage", response_model=UsageResponse)
def get_usage():
    """Get API usage metrics per tenant. Production: dashboard for billing."""
    return UsageResponse(
        tenants=usage_tracker.get_all_usage(),
        timestamp=datetime.now(),
    )


# ── Data Engineering Endpoints ────────────────────────────────


@app.post("/api/de/stream", response_model=DEStreamingResponse)
def de_stream_ingest(request: CommandCenterRequest):
    """Ingest patient data through event streaming pipeline with schema validation."""
    ep = "de_stream"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        from data_engineering.streaming import event_stream

        result = event_stream.ingest_patient_event(
            {
                "patient_id": request.patient_id,
                "heart_rate": request.heart_rate,
                "respiratory_rate": request.respiratory_rate,
                "body_temperature": request.body_temperature,
                "oxygen_saturation": request.oxygen_saturation,
                "systolic_bp": request.systolic_bp,
                "diastolic_bp": request.diastolic_bp,
                "age": request.age,
                "chief_complaint": request.chief_complaint,
                "clinical_note": request.clinical_note,
            }
        )
        evolution = event_stream.registry.get_evolution("patient_vitals")
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return DEStreamingResponse(
            patient_id=request.patient_id,
            events=result.get("events", []),
            dlq=result.get("dlq", []),
            metrics=result.get("metrics", {}),
            topic_stats=result.get("topic_stats", {}),
            schema_evolution=evolution,
        )
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/de/lineage", response_model=DELineageResponse)
def de_lineage():
    """Get column-level data lineage DAG for the full pipeline."""
    from data_engineering.lineage import DataLineageTracker

    tracker = DataLineageTracker()
    tracker.build_pipeline_lineage()
    dag = tracker.to_dag()
    impact = tracker.impact_analysis("raw_input", "heart_rate")
    pii = tracker.get_pii_columns()
    return DELineageResponse(dag=dag, impact_analysis=impact, pii_columns=pii)


@app.post("/api/de/quality", response_model=DEQualityResponse)
def de_quality(request: CommandCenterRequest):
    """Run data quality validation on patient data."""
    ep = "de_quality"
    REQUEST_COUNT.labels(endpoint=ep).inc()
    start = time.time()
    try:
        from data_engineering.quality import DataQualityFramework

        dq = DataQualityFramework()
        vitals_report = dq.validate_vitals(
            {
                "heart_rate": request.heart_rate,
                "respiratory_rate": request.respiratory_rate,
                "body_temperature": request.body_temperature,
                "oxygen_saturation": request.oxygen_saturation,
                "systolic_bp": request.systolic_bp,
                "diastolic_bp": request.diastolic_bp,
                "age": request.age,
            }
        )
        note_report = dq.validate_clinical_note(request.clinical_note)
        summary = dq.get_pipeline_quality_summary()
        REQUEST_LATENCY.labels(endpoint=ep).observe(time.time() - start)
        return DEQualityResponse(
            vitals_quality=vitals_report.to_dict(),
            note_quality=note_report.to_dict(),
            pipeline_summary=summary,
        )
    except Exception as e:
        REQUEST_FAILURES.labels(endpoint=ep).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/de/warehouse", response_model=DEWarehouseResponse)
def de_warehouse_stats():
    """Get analytics warehouse stats and recent encounters."""
    from data_engineering.warehouse import clinical_warehouse as wh

    wh.refresh_aggregates()
    return DEWarehouseResponse(
        encounter_id=0,
        warehouse_stats=wh.get_warehouse_stats(),
        risk_distribution=wh.query_risk_distribution(),
        recent_encounters=wh.query_encounters(limit=10),
    )


@app.get("/api/de/orchestrator", response_model=DEOrchestratorResponse)
def de_orchestrator():
    """Get ETL pipeline DAG definition and run history."""
    from data_engineering.orchestrator import build_hera_dag

    dag = build_hera_dag()
    return DEOrchestratorResponse(
        dag_definition=dag.get_dag_definition(),
        run_history=dag.get_run_history(),
    )


@app.get("/api/de/cdc", response_model=DECDCResponse)
def de_cdc_stream():
    """Get CDC event log and statistics."""
    from data_engineering.cdc import cdc_stream

    events = cdc_stream.replay()
    stats = cdc_stream.get_stats()
    return DECDCResponse(events=events, stats=stats)


@app.get("/api/de/catalog", response_model=DECatalogResponse)
def de_catalog():
    """Browse the data catalog with PII and freshness reports."""
    from data_engineering.catalog import data_catalog

    return DECatalogResponse(
        catalog=data_catalog.to_dict(),
        pii_report=data_catalog.get_pii_report(),
        freshness_report=data_catalog.get_freshness_report(),
    )


@app.get("/api/de/dashboard", response_model=DEDashboardResponse)
def de_dashboard():
    """Unified data engineering dashboard — all 7 systems at a glance."""
    from data_engineering.streaming import event_stream
    from data_engineering.lineage import DataLineageTracker
    from data_engineering.quality import DataQualityFramework
    from data_engineering.warehouse import clinical_warehouse
    from data_engineering.orchestrator import build_hera_dag
    from data_engineering.cdc import cdc_stream
    from data_engineering.catalog import data_catalog

    tracker = DataLineageTracker()
    tracker.build_pipeline_lineage()

    return DEDashboardResponse(
        streaming=event_stream.get_metrics(),
        quality=DataQualityFramework().get_pipeline_quality_summary(),
        lineage={
            "total_nodes": len(tracker._nodes),
            "total_edges": len(tracker._edges),
            "pii_columns": len(tracker.get_pii_columns()),
        },
        warehouse=clinical_warehouse.get_warehouse_stats(),
        orchestrator=build_hera_dag().get_dag_definition(),
        cdc=cdc_stream.get_stats(),
        catalog=data_catalog.to_dict(),
        timestamp=datetime.now(),
    )


@app.get("/api/de/live")
def de_live():
    """Live DE metrics — single endpoint for real-time dashboard updates."""
    from data_engineering.streaming import event_stream
    from data_engineering.warehouse import clinical_warehouse
    from data_engineering.cdc import cdc_stream
    from data_engineering.catalog import data_catalog

    clinical_warehouse.refresh_aggregates()

    return {
        "streaming": {
            **event_stream.get_metrics(),
            "topic_stats": event_stream.get_topic_stats(),
            "dlq": event_stream.get_dead_letter_queue()[-10:],
            "schema_evolution": event_stream.registry.get_evolution("patient_vitals"),
        },
        "warehouse": {
            "stats": clinical_warehouse.get_warehouse_stats(),
            "risk_distribution": clinical_warehouse.query_risk_distribution(),
            "recent_encounters": clinical_warehouse.query_encounters(limit=20),
            "hourly_summary": clinical_warehouse.query_hourly_summary(),
        },
        "cdc": {
            **cdc_stream.get_stats(),
            "recent_events": [
                {
                    "event_id": e.get("event_id", "") if isinstance(e, dict) else getattr(e, "event_id", ""),
                    "table": e.get("table", "") if isinstance(e, dict) else getattr(e, "table", ""),
                    "change_type": e.get("change_type", "") if isinstance(e, dict) else getattr(e, "change_type", ""),
                    "record_key": e.get("record_key", "") if isinstance(e, dict) else getattr(e, "record_key", ""),
                    "timestamp": e.get("timestamp", "") if isinstance(e, dict) else getattr(e, "timestamp", ""),
                }
                for e in cdc_stream.replay()[-20:]
            ],
        },
        "catalog": {
            "freshness": data_catalog.get_freshness_report(),
            "pii": data_catalog.get_pii_report(),
            "total_datasets": data_catalog.to_dict().get("total_datasets", 0),
            "total_columns": data_catalog.to_dict().get("total_columns", 0),
        },
        "timestamp": datetime.now().isoformat(),
    }
