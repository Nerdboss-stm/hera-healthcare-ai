"""Star schema analytics warehouse — dimensional model for clinical data.

Demonstrates: fact/dimension table design, star schema, SQLite-backed
OLAP warehouse, pre-computed aggregates, and analytical queries.
"""

from __future__ import annotations

import sqlite3
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ClinicalWarehouse:
    """SQLite-backed star schema warehouse for clinical analytics."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup_schema()

    def _setup_schema(self):
        """Create star schema: fact table + dimension tables."""
        cur = self._conn.cursor()

        # Dimension: Patient
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dim_patient (
                patient_key INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                age INTEGER,
                gender TEXT DEFAULT 'unknown',
                created_at TEXT
            )
        """)

        # Dimension: Diagnosis
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dim_diagnosis (
                diagnosis_key INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnosis_code TEXT,
                diagnosis_name TEXT,
                category TEXT,
                severity TEXT
            )
        """)

        # Dimension: Provider (system/agent)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dim_provider (
                provider_key INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_name TEXT,
                provider_type TEXT,
                system TEXT
            )
        """)

        # Dimension: Time
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dim_time (
                time_key INTEGER PRIMARY KEY AUTOINCREMENT,
                full_datetime TEXT,
                date TEXT,
                hour INTEGER,
                day_of_week TEXT,
                month INTEGER,
                year INTEGER
            )
        """)

        # Fact: Clinical Encounters
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fact_clinical_encounters (
                encounter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_key INTEGER REFERENCES dim_patient(patient_key),
                diagnosis_key INTEGER REFERENCES dim_diagnosis(diagnosis_key),
                provider_key INTEGER REFERENCES dim_provider(provider_key),
                time_key INTEGER REFERENCES dim_time(time_key),
                heart_rate REAL,
                respiratory_rate REAL,
                body_temperature REAL,
                oxygen_saturation REAL,
                systolic_bp REAL,
                diastolic_bp REAL,
                mean_arterial_pressure REAL,
                risk_score REAL,
                risk_prediction TEXT,
                confidence REAL,
                esi_level INTEGER,
                entity_count INTEGER,
                summary_compression REAL,
                safety_score REAL,
                safety_passed INTEGER,
                pipeline_latency_ms REAL,
                fhir_resource_count INTEGER,
                feedback_loops INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)

        # Aggregate: Hourly summary
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agg_hourly_summary (
                hour_key TEXT PRIMARY KEY,
                encounter_count INTEGER,
                avg_risk_score REAL,
                avg_latency_ms REAL,
                high_risk_count INTEGER,
                safety_failure_count INTEGER,
                avg_entity_count REAL,
                updated_at TEXT
            )
        """)

        # Indexes
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_patient ON fact_clinical_encounters(patient_key)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_risk ON fact_clinical_encounters(risk_score)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_time ON fact_clinical_encounters(time_key)"
        )

        # Seed dimension: providers (the HERA agents)
        providers = [
            ("Triage Agent", "agent", "multi_agent_reasoning"),
            ("Diagnostic Agent", "agent", "multi_agent_reasoning"),
            ("Treatment Agent", "agent", "multi_agent_reasoning"),
            ("NER Extractor", "system", "ner"),
            ("Risk Predictor", "model", "risk_prediction"),
            ("T5 Summarizer", "model", "summarization"),
            ("Safety Evaluator", "system", "evaluation"),
            ("RAG Pipeline", "system", "rag"),
            ("FHIR Converter", "system", "fhir"),
        ]
        for name, ptype, system in providers:
            cur.execute(
                "INSERT OR IGNORE INTO dim_provider (provider_name, provider_type, system) VALUES (?, ?, ?)",
                (name, ptype, system),
            )

        self._conn.commit()

    def _get_or_create_time(self, dt: datetime) -> int:
        cur = self._conn.cursor()
        full = dt.isoformat()
        cur.execute("SELECT time_key FROM dim_time WHERE full_datetime = ?", (full,))
        row = cur.fetchone()
        if row:
            return row["time_key"]
        cur.execute(
            "INSERT INTO dim_time (full_datetime, date, hour, day_of_week, month, year) VALUES (?, ?, ?, ?, ?, ?)",
            (
                full,
                dt.strftime("%Y-%m-%d"),
                dt.hour,
                dt.strftime("%A"),
                dt.month,
                dt.year,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def _get_or_create_patient(
        self, patient_id: str, age: int = 0, gender: str = "unknown"
    ) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT patient_key FROM dim_patient WHERE patient_id = ?", (patient_id,)
        )
        row = cur.fetchone()
        if row:
            return row["patient_key"]
        cur.execute(
            "INSERT INTO dim_patient (patient_id, age, gender, created_at) VALUES (?, ?, ?, ?)",
            (patient_id, age, gender, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return cur.lastrowid

    def _get_or_create_diagnosis(
        self, name: str, category: str = "", severity: str = ""
    ) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT diagnosis_key FROM dim_diagnosis WHERE diagnosis_name = ?", (name,)
        )
        row = cur.fetchone()
        if row:
            return row["diagnosis_key"]
        cur.execute(
            "INSERT INTO dim_diagnosis (diagnosis_name, category, severity) VALUES (?, ?, ?)",
            (name, category, severity),
        )
        self._conn.commit()
        return cur.lastrowid

    def load_encounter(self, pipeline_result: dict) -> int:
        """Load a complete pipeline result into the warehouse."""
        now = datetime.now(timezone.utc)
        patient_id = pipeline_result.get("patient_id", "unknown")

        patient_key = self._get_or_create_patient(
            patient_id,
            age=pipeline_result.get("age", 0),
            gender=pipeline_result.get("gender", "unknown"),
        )
        time_key = self._get_or_create_time(now)

        # Extract diagnosis from reasoning stage
        diagnosis_name = "Unknown"
        stages = pipeline_result.get("stages", [])
        for s in stages:
            if s.get("system") == "agents" and s.get("result"):
                dx = s["result"].get("diagnosis", {})
                diagnosis_name = dx.get("primary_diagnosis", "Unknown")
        diagnosis_key = self._get_or_create_diagnosis(diagnosis_name)

        # Extract metrics from stages
        ner_count = 0
        risk_score = 0.0
        risk_prediction = "Unknown"
        confidence = 0.0
        esi_level = 0
        compression = 0.0
        safety_score = 0.0
        safety_passed = 0
        fhir_count = 0

        for s in stages:
            sys = s.get("system", "")
            result = s.get("result", {})
            if sys == "ner":
                ner_count = result.get("entity_count", 0)
            elif sys == "risk_predictor":
                risk_score = result.get("risk_score", 0.0)
                risk_prediction = result.get("prediction", "Unknown")
                confidence = result.get("confidence", 0.0)
            elif sys == "agents":
                triage = result.get("triage", {})
                esi_level = triage.get("esi_level", 0)
            elif sys == "summarizer":
                compression = result.get("compression", 0.0)
            elif sys == "evaluator":
                safety_score = result.get("overall_score", 0.0)
                safety_passed = 1 if result.get("pass_threshold", False) else 0
            elif sys == "fhir":
                fhir_count = result.get("resource_count", 0)

        vitals = pipeline_result.get("vitals", {})
        sbp = vitals.get("systolic_bp", 0)
        dbp = vitals.get("diastolic_bp", 0)
        map_val = round(dbp + (sbp - dbp) / 3, 1) if sbp and dbp else 0

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO fact_clinical_encounters (
                patient_key, diagnosis_key, provider_key, time_key,
                heart_rate, respiratory_rate, body_temperature,
                oxygen_saturation, systolic_bp, diastolic_bp,
                mean_arterial_pressure, risk_score, risk_prediction,
                confidence, esi_level, entity_count, summary_compression,
                safety_score, safety_passed, pipeline_latency_ms,
                fhir_resource_count, feedback_loops, created_at
            ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                patient_key,
                diagnosis_key,
                time_key,
                vitals.get("heart_rate", 0),
                vitals.get("respiratory_rate", 0),
                vitals.get("body_temperature", 0),
                vitals.get("oxygen_saturation", 0),
                sbp,
                dbp,
                map_val,
                risk_score,
                risk_prediction,
                confidence,
                esi_level,
                ner_count,
                compression,
                safety_score,
                safety_passed,
                pipeline_result.get("overall_latency_ms", 0),
                fhir_count,
                pipeline_result.get("feedback_loops_triggered", 0),
                now.isoformat(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def refresh_aggregates(self):
        """Recompute hourly aggregates."""
        cur = self._conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO agg_hourly_summary (
                hour_key, encounter_count, avg_risk_score, avg_latency_ms,
                high_risk_count, safety_failure_count, avg_entity_count, updated_at
            )
            SELECT
                dt.date || 'T' || printf('%02d', dt.hour) AS hour_key,
                COUNT(*) AS encounter_count,
                ROUND(AVG(f.risk_score), 3) AS avg_risk_score,
                ROUND(AVG(f.pipeline_latency_ms), 1) AS avg_latency_ms,
                SUM(CASE WHEN f.risk_score > 0.7 THEN 1 ELSE 0 END) AS high_risk_count,
                SUM(CASE WHEN f.safety_passed = 0 THEN 1 ELSE 0 END) AS safety_failure_count,
                ROUND(AVG(f.entity_count), 1) AS avg_entity_count,
                datetime('now') AS updated_at
            FROM fact_clinical_encounters f
            JOIN dim_time dt ON f.time_key = dt.time_key
            GROUP BY dt.date, dt.hour
        """)
        self._conn.commit()

    def query_encounters(self, limit: int = 20) -> list[dict]:
        """Query recent encounters with full dimensional context."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT
                p.patient_id, p.age, p.gender,
                d.diagnosis_name, d.category,
                f.risk_score, f.risk_prediction, f.confidence,
                f.esi_level, f.entity_count, f.safety_score,
                f.safety_passed, f.pipeline_latency_ms,
                f.fhir_resource_count, f.feedback_loops,
                f.heart_rate, f.systolic_bp, f.diastolic_bp,
                dt.full_datetime
            FROM fact_clinical_encounters f
            JOIN dim_patient p ON f.patient_key = p.patient_key
            JOIN dim_diagnosis d ON f.diagnosis_key = d.diagnosis_key
            JOIN dim_time dt ON f.time_key = dt.time_key
            ORDER BY f.encounter_id DESC
            LIMIT ?
        """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def query_risk_distribution(self) -> dict:
        """Get risk score distribution for analytics."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT
                CASE
                    WHEN risk_score < 0.3 THEN 'low'
                    WHEN risk_score < 0.7 THEN 'medium'
                    ELSE 'high'
                END AS risk_level,
                COUNT(*) AS count,
                ROUND(AVG(pipeline_latency_ms), 1) AS avg_latency
            FROM fact_clinical_encounters
            GROUP BY risk_level
        """)
        return {
            row["risk_level"]: {
                "count": row["count"],
                "avg_latency": row["avg_latency"],
            }
            for row in cur.fetchall()
        }

    def get_warehouse_stats(self) -> dict:
        """Get warehouse metadata and stats."""
        cur = self._conn.cursor()
        tables = {}
        for tbl in [
            "dim_patient",
            "dim_diagnosis",
            "dim_provider",
            "dim_time",
            "fact_clinical_encounters",
            "agg_hourly_summary",
        ]:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM {tbl}")  # noqa: S608
            tables[tbl] = cur.fetchone()["cnt"]

        return {
            "tables": tables,
            "schema_type": "star_schema",
            "fact_table": "fact_clinical_encounters",
            "dimensions": ["dim_patient", "dim_diagnosis", "dim_provider", "dim_time"],
            "aggregates": ["agg_hourly_summary"],
        }
