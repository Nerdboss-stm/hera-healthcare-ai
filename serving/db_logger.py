import json
import sys
import psycopg2
from datetime import datetime
from config.settings import DB_CONFIG


def _get_conn():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"DB connection failed: {e}", file=sys.stderr, flush=True)
        return None


def _safe_json(obj):
    """Safely serialize to JSON, handling non-serializable types."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return json.dumps(str(obj))


def log_to_db(note: str, summary: str, status: str = "SUCCESS"):
    conn = _get_conn()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO summaries (note, summary, status, timestamp) VALUES (%s, %s, %s, %s)",
            (note, summary, status, datetime.now()),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Logging failed: {e}", file=sys.stderr, flush=True)


def log_prediction(vitals: dict, label: str, confidence: float):
    conn = _get_conn()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO patient_predictions
            (heart_rate, resp_rate, temperature, spo2, systolic_bp, diastolic_bp,
             age, bmi, map, predicted_label, confidence, predicted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                vitals.get("heart_rate"),
                vitals.get("respiratory_rate"),
                vitals.get("body_temperature"),
                vitals.get("oxygen_saturation"),
                vitals.get("systolic_bp"),
                vitals.get("diastolic_bp"),
                vitals.get("age"),
                vitals.get("calculated_bmi"),
                vitals.get("calculated_map"),
                label,
                float(confidence) if confidence is not None else None,
                datetime.now(),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Prediction logging failed: {e}", file=sys.stderr, flush=True)


def log_reasoning(
    patient_id: str,
    complaint: str,
    esi: int,
    diagnosis: str,
    confidence: float,
    disposition: str,
    consensus: float,
    latency: float,
    audit,
):
    conn = _get_conn()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO clinical_reasoning_sessions
            (patient_id, chief_complaint, esi_level, primary_diagnosis,
             diagnosis_confidence, disposition, consensus_score,
             pipeline_latency_ms, reasoning_audit, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                str(patient_id),
                str(complaint),
                int(esi),
                str(diagnosis),
                float(confidence) if confidence else 0.0,
                str(disposition),
                float(consensus) if consensus else 0.0,
                float(latency) if latency else 0.0,
                _safe_json(audit),
                datetime.now(),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[DB] Reasoning logged for {patient_id}", flush=True)
    except Exception as e:
        print(f"Reasoning logging failed: {e}", file=sys.stderr, flush=True)


def log_ner(
    patient_id: str,
    note_hash: str,
    count: int,
    medications: list,
    conditions: list,
    procedures: list,
    labs: list,
):
    conn = _get_conn()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO ner_extractions
            (patient_id, note_hash, entity_count, medications, conditions,
             procedures, lab_values, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                str(patient_id),
                str(note_hash),
                int(count),
                _safe_json(medications),
                _safe_json(conditions),
                _safe_json(procedures),
                _safe_json(labs),
                datetime.now(),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[DB] NER logged: {count} entities", flush=True)
    except Exception as e:
        print(f"NER logging failed: {e}", file=sys.stderr, flush=True)


def log_evaluation(
    note_hash: str,
    overall: float,
    passed: bool,
    factual: float,
    hallucination: float,
    accuracy: float,
    safety: bool,
    details: dict,
):
    conn = _get_conn()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO evaluation_reports
            (source_note_hash, overall_score, pass_threshold,
             factual_consistency_score, hallucination_score,
             medical_accuracy_score, clinical_safety_safe,
             details, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                str(note_hash),
                float(overall) if overall is not None else 0.0,
                bool(passed),
                float(factual) if factual is not None else 0.0,
                float(hallucination) if hallucination is not None else 0.0,
                float(accuracy) if accuracy is not None else 0.0,
                bool(safety),
                _safe_json(details),
                datetime.now(),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[DB] Evaluation logged: score={overall}", flush=True)
    except Exception as e:
        print(f"Evaluation logging failed: {e}", file=sys.stderr, flush=True)
