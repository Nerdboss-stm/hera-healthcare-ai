CREATE TABLE IF NOT EXISTS patient_predictions (
    id SERIAL PRIMARY KEY,
    heart_rate FLOAT,
    resp_rate FLOAT,
    temperature FLOAT,
    spo2 FLOAT,
    systolic_bp FLOAT,
    diastolic_bp FLOAT,
    age INT,
    bmi FLOAT,
    map FLOAT,
    predicted_label VARCHAR(20),
    confidence FLOAT,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    note TEXT NOT NULL,
    summary TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'SUCCESS',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Multi-Agent Clinical Reasoning audit trail
CREATE TABLE IF NOT EXISTS clinical_reasoning_sessions (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(100) NOT NULL,
    chief_complaint TEXT NOT NULL,
    esi_level INT,
    primary_diagnosis VARCHAR(200),
    diagnosis_confidence FLOAT,
    disposition VARCHAR(50),
    consensus_score FLOAT,
    pipeline_latency_ms FLOAT,
    reasoning_audit JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NER extraction results
CREATE TABLE IF NOT EXISTS ner_extractions (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(100),
    note_hash VARCHAR(64),
    entity_count INT,
    medications JSONB,
    conditions JSONB,
    procedures JSONB,
    lab_values JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation reports
CREATE TABLE IF NOT EXISTS evaluation_reports (
    id SERIAL PRIMARY KEY,
    source_note_hash VARCHAR(64),
    overall_score FLOAT,
    pass_threshold BOOLEAN,
    factual_consistency_score FLOAT,
    hallucination_score FLOAT,
    medical_accuracy_score FLOAT,
    clinical_safety_safe BOOLEAN,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
