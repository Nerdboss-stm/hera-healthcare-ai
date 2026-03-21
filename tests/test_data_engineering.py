"""Tests for the data engineering layer — all 7 modules."""


class TestEventStreaming:
    """Tests for the Kafka-style event streaming pipeline."""

    def test_schema_registry_defaults(self):
        from data_engineering.streaming import SchemaRegistry

        registry = SchemaRegistry()
        schemas = registry.list_schemas()
        assert "patient_vitals" in schemas
        assert schemas["patient_vitals"] == 3  # v1, v2, v3

    def test_schema_validation_pass(self):
        from data_engineering.streaming import SchemaRegistry

        registry = SchemaRegistry()
        schema = registry.get_version("patient_vitals", "1.0")
        valid, errors = schema.validate({"patient_id": "P-001", "heart_rate": 88.0})
        assert valid
        assert len(errors) == 0

    def test_schema_validation_fail(self):
        from data_engineering.streaming import SchemaRegistry

        registry = SchemaRegistry()
        schema = registry.get_version("patient_vitals", "1.0")
        valid, errors = schema.validate({"heart_rate": 88.0})  # missing patient_id
        assert not valid
        assert any("patient_id" in e for e in errors)

    def test_schema_evolution(self):
        from data_engineering.streaming import SchemaRegistry

        registry = SchemaRegistry()
        evolution = registry.get_evolution("patient_vitals")
        assert len(evolution) == 3
        assert evolution[0]["version"] == "1.0"
        assert evolution[2]["version"] == "3.0"
        assert "mean_arterial_pressure" in evolution[2]["fields"]

    def test_produce_and_consume(self):
        from data_engineering.streaming import EventStream

        stream = EventStream()
        evt = stream.produce("test-topic", "key-1", {"data": "hello"})
        assert evt is not None
        assert evt.topic == "test-topic"

        events = stream.consume("test-topic", "consumer-1")
        assert len(events) == 1
        assert events[0].value["data"] == "hello"

    def test_dead_letter_queue(self):
        from data_engineering.streaming import EventStream

        stream = EventStream()
        # Missing required field should go to DLQ
        result = stream.produce(
            "vitals", "P-001",
            {"heart_rate": 88.0},  # missing patient_id for v3
            schema_name="patient_vitals", schema_version="3.0",
        )
        assert result is None  # rejected
        dlq = stream.get_dead_letter_queue()
        assert len(dlq) >= 1

    def test_ingest_patient_event(self):
        from data_engineering.streaming import EventStream

        stream = EventStream()
        result = stream.ingest_patient_event({
            "patient_id": "P-001",
            "heart_rate": 88.0,
            "respiratory_rate": 18.0,
            "body_temperature": 37.0,
            "oxygen_saturation": 96.0,
            "systolic_bp": 142.0,
            "diastolic_bp": 88.0,
            "age": 65,
            "chief_complaint": "chest pain",
            "clinical_note": "65-year-old male with chest pain.",
        })
        assert result["patient_id"] == "P-001"
        assert len(result["events"]) >= 1
        assert result["metrics"]["events_produced"] >= 1


class TestDataLineage:
    """Tests for column-level data lineage tracking."""

    def test_build_pipeline_lineage(self):
        from data_engineering.lineage import DataLineageTracker

        tracker = DataLineageTracker()
        tracker.build_pipeline_lineage()
        dag = tracker.to_dag()
        assert dag["total_nodes"] > 20
        assert dag["total_edges"] > 20
        assert "ner" in dag["stages"]
        assert "fhir_export" in dag["stages"]

    def test_upstream_tracing(self):
        from data_engineering.lineage import DataLineageTracker

        tracker = DataLineageTracker()
        tracker.build_pipeline_lineage()
        upstream = tracker.get_upstream("risk_output", "risk_score")
        assert len(upstream) > 0
        sources = [u["source"] for u in upstream]
        assert any("heart_rate" in s for s in sources)

    def test_downstream_tracing(self):
        from data_engineering.lineage import DataLineageTracker

        tracker = DataLineageTracker()
        tracker.build_pipeline_lineage()
        downstream = tracker.get_downstream("raw_input", "heart_rate")
        assert len(downstream) > 0

    def test_pii_detection(self):
        from data_engineering.lineage import DataLineageTracker

        tracker = DataLineageTracker()
        tracker.build_pipeline_lineage()
        pii = tracker.get_pii_columns()
        assert len(pii) > 0
        fqns = [p["fqn"] for p in pii]
        assert "raw_input.patient_id" in fqns

    def test_impact_analysis(self):
        from data_engineering.lineage import DataLineageTracker

        tracker = DataLineageTracker()
        tracker.build_pipeline_lineage()
        impact = tracker.impact_analysis("raw_input", "heart_rate")
        assert impact["total_downstream"] > 0
        assert len(impact["affected_stages"]) > 0


class TestDataQuality:
    """Tests for the data quality framework."""

    def test_validate_good_vitals(self):
        from data_engineering.quality import DataQualityFramework

        dq = DataQualityFramework()
        report = dq.validate_vitals({
            "heart_rate": 88, "respiratory_rate": 18,
            "body_temperature": 37.0, "oxygen_saturation": 96,
            "systolic_bp": 120, "diastolic_bp": 80, "age": 65,
        })
        assert report.score > 0.9
        assert all(c.passed for c in report.checks)

    def test_validate_bad_vitals(self):
        from data_engineering.quality import DataQualityFramework

        dq = DataQualityFramework()
        report = dq.validate_vitals({
            "heart_rate": 300,  # out of range
            "systolic_bp": 50, "diastolic_bp": 200,  # inverted
        })
        assert report.score < 0.8
        failed = [c for c in report.checks if not c.passed]
        assert len(failed) > 0

    def test_validate_clinical_note(self):
        from data_engineering.quality import DataQualityFramework

        dq = DataQualityFramework()
        report = dq.validate_clinical_note(
            "Patient presenting with chest pain and history of hypertension."
        )
        assert report.score > 0.8

    def test_validate_note_with_ssn(self):
        from data_engineering.quality import DataQualityFramework

        dq = DataQualityFramework()
        report = dq.validate_clinical_note(
            "Patient John Doe SSN 123-45-6789 presenting with chest pain."
        )
        ssn_check = [c for c in report.checks if c.name == "no_embedded_ssn"]
        assert len(ssn_check) == 1
        assert not ssn_check[0].passed

    def test_pipeline_quality_summary(self):
        from data_engineering.quality import DataQualityFramework

        dq = DataQualityFramework()
        dq.validate_vitals({"heart_rate": 88, "respiratory_rate": 18,
                            "body_temperature": 37.0, "oxygen_saturation": 96,
                            "systolic_bp": 120, "diastolic_bp": 80})
        summary = dq.get_pipeline_quality_summary()
        assert "overall_avg" in summary
        assert summary["total_checks_run"] > 0


class TestWarehouse:
    """Tests for the star schema analytics warehouse."""

    def test_warehouse_creation(self):
        from data_engineering.warehouse import ClinicalWarehouse

        wh = ClinicalWarehouse()
        stats = wh.get_warehouse_stats()
        assert "dim_patient" in stats["tables"]
        assert "fact_clinical_encounters" in stats["tables"]
        assert stats["schema_type"] == "star_schema"

    def test_load_encounter(self):
        from data_engineering.warehouse import ClinicalWarehouse

        wh = ClinicalWarehouse()
        enc_id = wh.load_encounter({
            "patient_id": "P-001",
            "age": 65,
            "gender": "male",
            "vitals": {"heart_rate": 88, "systolic_bp": 140, "diastolic_bp": 90},
            "stages": [
                {"system": "ner", "result": {"entity_count": 5}},
                {"system": "risk_predictor", "result": {"risk_score": 0.75, "prediction": "High Risk", "confidence": 0.85}},
                {"system": "agents", "result": {"triage": {"esi_level": 2}, "diagnosis": {"primary_diagnosis": "STEMI"}}},
            ],
            "overall_latency_ms": 5000,
        })
        assert enc_id > 0

    def test_query_encounters(self):
        from data_engineering.warehouse import ClinicalWarehouse

        wh = ClinicalWarehouse()
        wh.load_encounter({
            "patient_id": "P-002",
            "age": 45,
            "vitals": {"heart_rate": 72, "systolic_bp": 118, "diastolic_bp": 76},
            "stages": [],
        })
        encounters = wh.query_encounters()
        assert len(encounters) >= 1

    def test_dimension_providers_seeded(self):
        from data_engineering.warehouse import ClinicalWarehouse

        wh = ClinicalWarehouse()
        stats = wh.get_warehouse_stats()
        assert stats["tables"]["dim_provider"] == 9  # 9 HERA agents


class TestOrchestrator:
    """Tests for the Airflow-style ETL orchestrator."""

    def test_build_hera_dag(self):
        from data_engineering.orchestrator import build_hera_dag

        dag = build_hera_dag()
        defn = dag.get_dag_definition()
        assert defn["total_tasks"] == 16
        assert len(defn["validation_errors"]) == 0

    def test_topological_sort(self):
        from data_engineering.orchestrator import build_hera_dag

        dag = build_hera_dag()
        order = dag.get_execution_order()
        assert order.index("ingest_data") < order.index("validate_quality")
        assert order.index("validate_quality") < order.index("extract_entities")
        assert order.index("export_fhir") < order.index("load_warehouse")

    def test_dag_execution(self):
        from data_engineering.orchestrator import PipelineDAG, TaskDefinition

        dag = PipelineDAG("test")
        dag.add_task(
            TaskDefinition(task_id="step1", callable_name="test"),
            callable_fn=lambda ctx: "done",
        )
        dag.add_task(
            TaskDefinition(task_id="step2", callable_name="test", depends_on=["step1"]),
            callable_fn=lambda ctx: "done2",
        )
        run = dag.execute()
        assert run.state == "success"
        assert run.task_runs["step1"].state.value == "success"
        assert run.task_runs["step2"].state.value == "success"

    def test_upstream_failure_propagation(self):
        from data_engineering.orchestrator import PipelineDAG, TaskDefinition

        dag = PipelineDAG("test")
        dag.add_task(
            TaskDefinition(task_id="fail_step", callable_name="test", retries=0),
            callable_fn=lambda ctx: (_ for _ in ()).throw(ValueError("boom")),
        )
        dag.add_task(
            TaskDefinition(task_id="dep_step", callable_name="test", depends_on=["fail_step"]),
            callable_fn=lambda ctx: "ok",
        )
        run = dag.execute()
        assert run.state == "failed"
        assert run.task_runs["dep_step"].state.value == "upstream_failed"


class TestCDC:
    """Tests for the Change Data Capture stream."""

    def test_capture_insert(self):
        from data_engineering.cdc import CDCStream, ChangeType

        cdc = CDCStream()
        evt = cdc.capture_change("patients", "P-001", ChangeType.INSERT,
                                 new_state={"name": "John", "age": 65})
        assert evt.change_type == "INSERT"
        assert evt.before is None
        assert evt.after["name"] == "John"

    def test_capture_update_with_diff(self):
        from data_engineering.cdc import CDCStream, ChangeType

        cdc = CDCStream()
        cdc.capture_change("patients", "P-001", ChangeType.INSERT,
                           new_state={"name": "John", "age": 65})
        evt = cdc.capture_change("patients", "P-001", ChangeType.UPDATE,
                                 new_state={"name": "John", "age": 66})
        assert evt.before["age"] == 65
        assert evt.after["age"] == 66
        assert "age" in evt.diff

    def test_replay_events(self):
        from data_engineering.cdc import CDCStream, ChangeType

        cdc = CDCStream()
        cdc.capture_change("t1", "k1", ChangeType.INSERT, {"val": 1})
        cdc.capture_change("t1", "k2", ChangeType.INSERT, {"val": 2})
        cdc.capture_change("t2", "k3", ChangeType.INSERT, {"val": 3})

        all_events = cdc.replay()
        assert len(all_events) == 3

        t1_events = cdc.replay(table="t1")
        assert len(t1_events) == 2

    def test_consumer_offsets(self):
        from data_engineering.cdc import CDCStream, ChangeType

        cdc = CDCStream()
        cdc.capture_change("t", "k1", ChangeType.INSERT, {"v": 1})
        cdc.capture_change("t", "k2", ChangeType.INSERT, {"v": 2})

        batch1 = cdc.consume("consumer-a", max_events=1)
        assert len(batch1) == 1

        batch2 = cdc.consume("consumer-a", max_events=10)
        assert len(batch2) == 1  # should get remaining event

    def test_capture_encounter(self):
        from data_engineering.cdc import CDCStream

        cdc = CDCStream()
        events = cdc.capture_encounter({
            "patient_id": "P-001",
            "age": 65,
            "heart_rate": 88,
            "systolic_bp": 140,
            "diastolic_bp": 90,
            "stages": [{"system": "ner", "status": "completed", "latency_ms": 50}],
        })
        assert len(events) >= 2  # patient + vitals + stages
        stats = cdc.get_stats()
        assert stats["total_events"] > 0


class TestDataCatalog:
    """Tests for the data catalog."""

    def test_default_datasets(self):
        from data_engineering.catalog import DataCatalog

        cat = DataCatalog()
        info = cat.to_dict()
        assert info["total_datasets"] == 12
        assert info["total_columns"] > 50

    def test_search(self):
        from data_engineering.catalog import DataCatalog

        cat = DataCatalog()
        results = cat.search("risk")
        assert len(results) > 0
        assert any("risk" in r["name"].lower() for r in results)

    def test_pii_report(self):
        from data_engineering.catalog import DataCatalog

        cat = DataCatalog()
        pii = cat.get_pii_report()
        assert len(pii) > 0
        assert any(p["column"] == "patient_id" for p in pii)

    def test_freshness_update(self):
        from data_engineering.catalog import DataCatalog

        cat = DataCatalog()
        cat.update_freshness("raw_patient_vitals", row_count=100)
        ds = cat.get("raw_patient_vitals")
        assert ds.row_count == 100

    def test_freshness_report(self):
        from data_engineering.catalog import DataCatalog

        cat = DataCatalog()
        report = cat.get_freshness_report()
        assert len(report) == 12


class TestDEAPIEndpoints:
    """Tests for the data engineering API endpoints."""

    def test_lineage_endpoint(self):
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/de/lineage")
        assert response.status_code == 200
        data = response.json()
        assert "dag" in data
        assert data["dag"]["total_nodes"] > 0

    def test_orchestrator_endpoint(self):
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/de/orchestrator")
        assert response.status_code == 200
        data = response.json()
        assert data["dag_definition"]["total_tasks"] == 16

    def test_catalog_endpoint(self):
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/de/catalog")
        assert response.status_code == 200
        data = response.json()
        assert data["catalog"]["total_datasets"] == 12

    def test_cdc_endpoint(self):
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/de/cdc")
        assert response.status_code == 200
        assert "stats" in response.json()

    def test_dashboard_endpoint(self):
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/de/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "streaming" in data
        assert "warehouse" in data
        assert "catalog" in data

    def test_health_includes_de_services(self):
        from serving.api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/health")
        data = response.json()
        assert data["version"] == "4.0.0"
        assert "event_streaming" in data["services"]
        assert "data_catalog" in data["services"]
        assert len(data["services"]) == 17  # 10 original + 7 DE
