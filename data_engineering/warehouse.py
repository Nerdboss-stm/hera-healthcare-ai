"""Star schema analytics warehouse — PostgreSQL-backed dimensional model.

Uses the shared hera_ai PostgreSQL database for persistent storage.
Falls back to in-memory SQLite when PostgreSQL is unavailable (local dev).
"""

from __future__ import annotations

import sys
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _connect_pg():
    """Try to connect to PostgreSQL. Returns (conn, is_pg) or (None, False)."""
    try:
        import psycopg2
        import psycopg2.extras
        from config.settings import DB_CONFIG

        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        return conn, True
    except Exception as e:
        logger.info("PostgreSQL unavailable (%s), using SQLite fallback", e)
        return None, False


def _connect_sqlite():
    """Create in-memory SQLite connection as fallback."""
    import sqlite3

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


class ClinicalWarehouse:
    """Star schema warehouse backed by PostgreSQL (or SQLite fallback).

    Args:
        force_sqlite: If True, skip PostgreSQL and use in-memory SQLite.
                      Useful for testing without a database dependency.
    """

    def __init__(self, force_sqlite: bool = False):
        if not force_sqlite:
            pg_conn, is_pg = _connect_pg()
        else:
            pg_conn, is_pg = None, False

        if is_pg and pg_conn:
            self._conn = pg_conn
            self._is_pg = True
            self._ph = "%s"  # placeholder style
        else:
            self._conn = _connect_sqlite()
            self._is_pg = False
            self._ph = "?"
            self._setup_sqlite_schema()

        if self._is_pg:
            self._seed_providers_pg()

    @property
    def backend(self) -> str:
        return "postgresql" if self._is_pg else "sqlite"

    def _reconnect_pg(self):
        """Reconnect to PostgreSQL if the connection was lost."""
        if not self._is_pg:
            return
        try:
            self._conn.close()
        except Exception:
            pass
        pg_conn, is_pg = _connect_pg()
        if is_pg and pg_conn:
            self._conn = pg_conn
        else:
            logger.warning("PostgreSQL reconnect failed, falling back to SQLite")
            self._conn = _connect_sqlite()
            self._is_pg = False
            self._ph = "?"
            self._setup_sqlite_schema()

    def _execute(self, sql: str, params: tuple = (), fetch: str = "none"):
        """Execute SQL with auto-reconnect for PostgreSQL."""
        try:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            if fetch == "one":
                row = cur.fetchone()
                if self._is_pg:
                    if row and cur.description:
                        cols = [d[0] for d in cur.description]
                        return dict(zip(cols, row))
                    return None
                return dict(row) if row else None
            elif fetch == "all":
                rows = cur.fetchall()
                if self._is_pg:
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, r)) for r in rows]
                return [dict(r) for r in rows]
            else:
                return cur
        except Exception as e:
            if self._is_pg and "connection" in str(e).lower():
                self._reconnect_pg()
                return self._execute(sql, params, fetch)
            raise

    def _commit(self):
        self._conn.commit()

    def _seed_providers_pg(self):
        """Seed providers into PostgreSQL if not present.

        If PG tables don't exist yet (schema not applied), falls back to SQLite.
        """
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM dim_provider")
            count = cur.fetchone()[0]
            if count == 0:
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
                        "INSERT INTO dim_provider (provider_name, provider_type, system) "
                        "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (name, ptype, system),
                    )
                self._commit()
        except Exception as e:
            logger.warning(
                "PG warehouse tables not found, falling back to SQLite: %s", e
            )
            try:
                self._conn.rollback()
                self._conn.close()
            except Exception:
                pass
            # Fall back to SQLite
            self._conn = _connect_sqlite()
            self._is_pg = False
            self._ph = "?"
            self._setup_sqlite_schema()

    def _setup_sqlite_schema(self):
        """Create star schema in SQLite (fallback mode)."""
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dim_patient (
                patient_key INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                age INTEGER, gender TEXT DEFAULT 'unknown',
                created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dim_diagnosis (
                diagnosis_key INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnosis_code TEXT, diagnosis_name TEXT,
                category TEXT, severity TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dim_provider (
                provider_key INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_name TEXT, provider_type TEXT, system TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dim_time (
                time_key INTEGER PRIMARY KEY AUTOINCREMENT,
                full_datetime TEXT, date TEXT, hour INTEGER,
                day_of_week TEXT, month INTEGER, year INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fact_clinical_encounters (
                encounter_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_key INTEGER, diagnosis_key INTEGER,
                provider_key INTEGER, time_key INTEGER,
                heart_rate REAL, respiratory_rate REAL, body_temperature REAL,
                oxygen_saturation REAL, systolic_bp REAL, diastolic_bp REAL,
                mean_arterial_pressure REAL, risk_score REAL,
                risk_prediction TEXT, risk_level TEXT,
                confidence REAL, esi_level INTEGER,
                entity_count INTEGER, summary_compression REAL,
                safety_score REAL, safety_passed INTEGER,
                pipeline_latency_ms REAL, fhir_resource_count INTEGER,
                feedback_loops INTEGER DEFAULT 0, created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agg_hourly_summary (
                hour_key TEXT PRIMARY KEY, encounter_count INTEGER,
                avg_risk_score REAL, avg_latency_ms REAL,
                high_risk_count INTEGER, safety_failure_count INTEGER,
                avg_entity_count REAL, updated_at TEXT
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_patient ON fact_clinical_encounters(patient_key)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_risk ON fact_clinical_encounters(risk_score)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_time ON fact_clinical_encounters(time_key)"
        )
        # Seed providers
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
        ph = self._ph
        full = dt.isoformat()
        row = self._execute(
            f"SELECT time_key FROM dim_time WHERE full_datetime = {ph}",
            (full,),
            fetch="one",
        )
        if row:
            return row["time_key"]
        cur = self._execute(
            f"INSERT INTO dim_time (full_datetime, date, hour, day_of_week, month, year) "
            f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})",
            (
                full,
                dt.strftime("%Y-%m-%d"),
                dt.hour,
                dt.strftime("%A"),
                dt.month,
                dt.year,
            ),
        )
        self._commit()
        if self._is_pg:
            row = self._execute(
                f"SELECT time_key FROM dim_time WHERE full_datetime = {ph}",
                (full,),
                fetch="one",
            )
            return row["time_key"] if row else 0
        return cur.lastrowid

    def _get_or_create_patient(
        self, patient_id: str, age: int = 0, gender: str = "unknown"
    ) -> int:
        ph = self._ph
        row = self._execute(
            f"SELECT patient_key FROM dim_patient WHERE patient_id = {ph}",
            (patient_id,),
            fetch="one",
        )
        if row:
            return row["patient_key"]
        now = datetime.now(timezone.utc).isoformat()
        if self._is_pg:
            self._execute(
                f"INSERT INTO dim_patient (patient_id, age, gender, created_at) "
                f"VALUES ({ph}, {ph}, {ph}, {ph}) ON CONFLICT (patient_id) DO NOTHING",
                (patient_id, age, gender, now),
            )
            self._commit()
            row = self._execute(
                f"SELECT patient_key FROM dim_patient WHERE patient_id = {ph}",
                (patient_id,),
                fetch="one",
            )
            return row["patient_key"] if row else 0
        else:
            cur = self._execute(
                f"INSERT OR IGNORE INTO dim_patient (patient_id, age, gender, created_at) "
                f"VALUES ({ph}, {ph}, {ph}, {ph})",
                (patient_id, age, gender, now),
            )
            self._commit()
            return cur.lastrowid or 0

    def _get_or_create_diagnosis(
        self, name: str, category: str = "", severity: str = ""
    ) -> int:
        ph = self._ph
        row = self._execute(
            f"SELECT diagnosis_key FROM dim_diagnosis WHERE diagnosis_name = {ph}",
            (name,),
            fetch="one",
        )
        if row:
            return row["diagnosis_key"]
        self._execute(
            f"INSERT INTO dim_diagnosis (diagnosis_name, category, severity) VALUES ({ph}, {ph}, {ph})",
            (name, category, severity),
        )
        self._commit()
        row = self._execute(
            f"SELECT diagnosis_key FROM dim_diagnosis WHERE diagnosis_name = {ph}",
            (name,),
            fetch="one",
        )
        return row["diagnosis_key"] if row else 0

    def load_encounter(self, pipeline_result: dict) -> int:
        """Load a complete pipeline result into the warehouse."""
        ph = self._ph
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
        risk_level = "Unknown"
        confidence = 0.0
        esi_level = 0
        compression = 0.0
        safety_score = 0.0
        safety_passed = False
        fhir_count = 0

        for s in stages:
            sys_name = s.get("system", "")
            result = s.get("result", {})
            if sys_name == "ner":
                ner_count = result.get("entity_count", 0)
            elif sys_name == "risk_predictor":
                risk_score = result.get("risk_score", 0.0)
                risk_prediction = result.get("prediction", "Unknown")
                risk_level = result.get("risk_level", risk_prediction)
                confidence = result.get("confidence", 0.0)
            elif sys_name == "agents":
                triage = result.get("triage", {})
                esi_level = triage.get("esi_level", 0)
            elif sys_name == "summarizer":
                compression = result.get("compression", 0.0)
            elif sys_name == "evaluator":
                safety_score = result.get("overall_score", 0.0)
                safety_passed = bool(result.get("pass_threshold", False))
            elif sys_name == "fhir":
                fhir_count = result.get("resource_count", 0)

        vitals = pipeline_result.get("vitals", {})
        sbp = vitals.get("systolic_bp", 0)
        dbp = vitals.get("diastolic_bp", 0)
        map_val = round(dbp + (sbp - dbp) / 3, 1) if sbp and dbp else 0

        phs24 = ", ".join([ph] * 24)
        if self._is_pg:
            self._execute(
                f"""INSERT INTO fact_clinical_encounters (
                    patient_key, diagnosis_key, provider_key, time_key,
                    heart_rate, respiratory_rate, body_temperature,
                    oxygen_saturation, systolic_bp, diastolic_bp,
                    mean_arterial_pressure, risk_score, risk_prediction,
                    risk_level, confidence, esi_level, entity_count,
                    summary_compression, safety_score, safety_passed,
                    pipeline_latency_ms, fhir_resource_count,
                    feedback_loops, created_at
                ) VALUES ({phs24})""",
                (
                    patient_key,
                    diagnosis_key,
                    1,
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
                    risk_level,
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
            self._commit()
            row = self._execute(
                "SELECT MAX(encounter_id) AS eid FROM fact_clinical_encounters",
                fetch="one",
            )
            return row["eid"] if row else 0
        else:
            # SQLite: safety_passed as int
            cur = self._execute(
                f"""INSERT INTO fact_clinical_encounters (
                    patient_key, diagnosis_key, provider_key, time_key,
                    heart_rate, respiratory_rate, body_temperature,
                    oxygen_saturation, systolic_bp, diastolic_bp,
                    mean_arterial_pressure, risk_score, risk_prediction,
                    risk_level, confidence, esi_level, entity_count,
                    summary_compression, safety_score, safety_passed,
                    pipeline_latency_ms, fhir_resource_count,
                    feedback_loops, created_at
                ) VALUES ({phs24})""",
                (
                    patient_key,
                    diagnosis_key,
                    1,
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
                    risk_level,
                    confidence,
                    esi_level,
                    ner_count,
                    compression,
                    safety_score,
                    1 if safety_passed else 0,
                    pipeline_result.get("overall_latency_ms", 0),
                    fhir_count,
                    pipeline_result.get("feedback_loops_triggered", 0),
                    now.isoformat(),
                ),
            )
            self._commit()
            return cur.lastrowid

    def refresh_aggregates(self):
        """Recompute hourly aggregates."""
        if self._is_pg:
            self._execute("""
                INSERT INTO agg_hourly_summary (
                    hour_key, encounter_count, avg_risk_score, avg_latency_ms,
                    high_risk_count, safety_failure_count, avg_entity_count, updated_at
                )
                SELECT
                    TO_CHAR(dt.full_datetime, 'YYYY-MM-DD"T"HH24') AS hour_key,
                    COUNT(*) AS encounter_count,
                    ROUND(AVG(f.risk_score)::numeric, 3) AS avg_risk_score,
                    ROUND(AVG(f.pipeline_latency_ms)::numeric, 1) AS avg_latency_ms,
                    SUM(CASE WHEN f.risk_score > 0.7 THEN 1 ELSE 0 END) AS high_risk_count,
                    SUM(CASE WHEN f.safety_passed = FALSE THEN 1 ELSE 0 END) AS safety_failure_count,
                    ROUND(AVG(f.entity_count)::numeric, 1) AS avg_entity_count,
                    NOW() AS updated_at
                FROM fact_clinical_encounters f
                JOIN dim_time dt ON f.time_key = dt.time_key
                GROUP BY TO_CHAR(dt.full_datetime, 'YYYY-MM-DD"T"HH24')
                ON CONFLICT (hour_key) DO UPDATE SET
                    encounter_count = EXCLUDED.encounter_count,
                    avg_risk_score = EXCLUDED.avg_risk_score,
                    avg_latency_ms = EXCLUDED.avg_latency_ms,
                    high_risk_count = EXCLUDED.high_risk_count,
                    safety_failure_count = EXCLUDED.safety_failure_count,
                    avg_entity_count = EXCLUDED.avg_entity_count,
                    updated_at = EXCLUDED.updated_at
            """)
        else:
            self._execute("""
                INSERT OR REPLACE INTO agg_hourly_summary (
                    hour_key, encounter_count, avg_risk_score, avg_latency_ms,
                    high_risk_count, safety_failure_count, avg_entity_count, updated_at
                )
                SELECT
                    dt.date || 'T' || printf('%02d', dt.hour) AS hour_key,
                    COUNT(*), ROUND(AVG(f.risk_score), 3),
                    ROUND(AVG(f.pipeline_latency_ms), 1),
                    SUM(CASE WHEN f.risk_score > 0.7 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN f.safety_passed = 0 THEN 1 ELSE 0 END),
                    ROUND(AVG(f.entity_count), 1), datetime('now')
                FROM fact_clinical_encounters f
                JOIN dim_time dt ON f.time_key = dt.time_key
                GROUP BY dt.date, dt.hour
            """)
        self._commit()

    def query_encounters(self, limit: int = 20) -> list[dict]:
        """Query recent encounters with full dimensional context."""
        return self._execute(
            f"""SELECT
                p.patient_id, p.age, p.gender,
                d.diagnosis_name, d.category,
                f.risk_score, f.risk_prediction, f.confidence,
                f.esi_level, f.entity_count, f.safety_score,
                f.safety_passed, f.pipeline_latency_ms,
                f.fhir_resource_count, f.feedback_loops,
                f.heart_rate, f.systolic_bp, f.diastolic_bp,
                f.created_at
            FROM fact_clinical_encounters f
            JOIN dim_patient p ON f.patient_key = p.patient_key
            JOIN dim_diagnosis d ON f.diagnosis_key = d.diagnosis_key
            JOIN dim_time dt ON f.time_key = dt.time_key
            ORDER BY f.encounter_id DESC
            LIMIT {self._ph}""",
            (limit,),
            fetch="all",
        )

    def query_risk_distribution(self) -> dict:
        """Get risk score distribution for analytics (4 levels)."""
        rows = self._execute(
            """
            SELECT
                CASE
                    WHEN risk_score >= 0.75 THEN 'critical'
                    WHEN risk_score >= 0.50 THEN 'high'
                    WHEN risk_score >= 0.25 THEN 'medium'
                    ELSE 'low'
                END AS risk_level,
                COUNT(*) AS cnt,
                ROUND(AVG(pipeline_latency_ms)::numeric, 1) AS avg_latency
            FROM fact_clinical_encounters
            GROUP BY 1
        """
            if self._is_pg
            else """
            SELECT
                CASE
                    WHEN risk_score >= 0.75 THEN 'critical'
                    WHEN risk_score >= 0.50 THEN 'high'
                    WHEN risk_score >= 0.25 THEN 'medium'
                    ELSE 'low'
                END AS risk_level,
                COUNT(*) AS cnt,
                ROUND(AVG(pipeline_latency_ms), 1) AS avg_latency
            FROM fact_clinical_encounters
            GROUP BY 1
        """,
            fetch="all",
        )
        return {
            row["risk_level"]: {"count": row["cnt"], "avg_latency": row["avg_latency"]}
            for row in rows
        }

    def query_hourly_summary(self, limit: int = 24) -> list[dict]:
        """Get recent hourly aggregate summaries."""
        return self._execute(
            f"SELECT * FROM agg_hourly_summary ORDER BY hour_key DESC LIMIT {self._ph}",
            (limit,),
            fetch="all",
        )

    def get_warehouse_stats(self) -> dict:
        """Get warehouse metadata and row counts."""
        tables = {}
        for tbl in [
            "dim_patient",
            "dim_diagnosis",
            "dim_provider",
            "dim_time",
            "fact_clinical_encounters",
            "agg_hourly_summary",
        ]:
            row = self._execute(
                f"SELECT COUNT(*) AS cnt FROM {tbl}",  # noqa: S608
                fetch="one",
            )
            tables[tbl] = row["cnt"] if row else 0

        return {
            "tables": tables,
            "schema_type": "star_schema",
            "backend": self.backend,
            "fact_table": "fact_clinical_encounters",
            "dimensions": ["dim_patient", "dim_diagnosis", "dim_provider", "dim_time"],
            "aggregates": ["agg_hourly_summary"],
        }


# Singleton — shared across the app
clinical_warehouse = ClinicalWarehouse()
print(
    f"[Warehouse] Backend: {clinical_warehouse.backend}",
    file=sys.stderr,
    flush=True,
)
