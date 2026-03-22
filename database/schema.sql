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

-- ══════════════════════════════════════════════════════════════
-- Data Engineering: Star Schema Analytics Warehouse
-- ══════════════════════════════════════════════════════════════

-- Dimension: Patient
CREATE TABLE IF NOT EXISTS dim_patient (
    patient_key SERIAL PRIMARY KEY,
    patient_id VARCHAR(100) UNIQUE NOT NULL,
    age INT,
    gender VARCHAR(20) DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension: Diagnosis
CREATE TABLE IF NOT EXISTS dim_diagnosis (
    diagnosis_key SERIAL PRIMARY KEY,
    diagnosis_code VARCHAR(20),
    diagnosis_name VARCHAR(200),
    category VARCHAR(100),
    severity VARCHAR(50)
);

-- Dimension: Provider (HERA agents/systems)
CREATE TABLE IF NOT EXISTS dim_provider (
    provider_key SERIAL PRIMARY KEY,
    provider_name VARCHAR(100),
    provider_type VARCHAR(50),
    system VARCHAR(100)
);

-- Dimension: Time
CREATE TABLE IF NOT EXISTS dim_time (
    time_key SERIAL PRIMARY KEY,
    full_datetime TIMESTAMP,
    date DATE,
    hour INT,
    day_of_week VARCHAR(10),
    month INT,
    year INT
);

-- Fact: Clinical Encounters (star schema center)
CREATE TABLE IF NOT EXISTS fact_clinical_encounters (
    encounter_id SERIAL PRIMARY KEY,
    patient_key INT REFERENCES dim_patient(patient_key),
    diagnosis_key INT REFERENCES dim_diagnosis(diagnosis_key),
    provider_key INT REFERENCES dim_provider(provider_key),
    time_key INT REFERENCES dim_time(time_key),
    heart_rate FLOAT,
    respiratory_rate FLOAT,
    body_temperature FLOAT,
    oxygen_saturation FLOAT,
    systolic_bp FLOAT,
    diastolic_bp FLOAT,
    mean_arterial_pressure FLOAT,
    risk_score FLOAT,
    risk_prediction VARCHAR(50),
    risk_level VARCHAR(50),
    confidence FLOAT,
    esi_level INT,
    entity_count INT,
    summary_compression FLOAT,
    safety_score FLOAT,
    safety_passed BOOLEAN DEFAULT FALSE,
    pipeline_latency_ms FLOAT,
    fhir_resource_count INT,
    feedback_loops INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aggregate: Hourly summary (pre-computed rollups)
CREATE TABLE IF NOT EXISTS agg_hourly_summary (
    hour_key VARCHAR(20) PRIMARY KEY,
    encounter_count INT,
    avg_risk_score FLOAT,
    avg_latency_ms FLOAT,
    high_risk_count INT,
    safety_failure_count INT,
    avg_entity_count FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data Engineering: CDC event log (persistent)
CREATE TABLE IF NOT EXISTS de_cdc_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(100),
    table_name VARCHAR(100),
    record_key VARCHAR(100),
    change_type VARCHAR(10),
    before_state JSONB,
    after_state JSONB,
    diff JSONB,
    checksum VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data Engineering: Streaming event log
CREATE TABLE IF NOT EXISTS de_stream_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(100),
    topic VARCHAR(100),
    partition_num INT,
    offset_num INT,
    key VARCHAR(100),
    value JSONB,
    schema_version VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data Engineering: Quality check history
CREATE TABLE IF NOT EXISTS de_quality_checks (
    id SERIAL PRIMARY KEY,
    check_type VARCHAR(50),
    dataset VARCHAR(100),
    score FLOAT,
    checks_passed INT,
    checks_failed INT,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for warehouse queries
CREATE INDEX IF NOT EXISTS idx_fact_patient ON fact_clinical_encounters(patient_key);
CREATE INDEX IF NOT EXISTS idx_fact_risk ON fact_clinical_encounters(risk_score);
CREATE INDEX IF NOT EXISTS idx_fact_time ON fact_clinical_encounters(time_key);
CREATE INDEX IF NOT EXISTS idx_fact_created ON fact_clinical_encounters(created_at);
CREATE INDEX IF NOT EXISTS idx_cdc_table ON de_cdc_events(table_name);
CREATE INDEX IF NOT EXISTS idx_cdc_created ON de_cdc_events(created_at);
CREATE INDEX IF NOT EXISTS idx_stream_topic ON de_stream_events(topic);
CREATE INDEX IF NOT EXISTS idx_quality_created ON de_quality_checks(created_at);

-- Seed dimension: HERA providers/agents
INSERT INTO dim_provider (provider_name, provider_type, system) VALUES
    ('Triage Agent', 'agent', 'multi_agent_reasoning'),
    ('Diagnostic Agent', 'agent', 'multi_agent_reasoning'),
    ('Treatment Agent', 'agent', 'multi_agent_reasoning'),
    ('NER Extractor', 'system', 'ner'),
    ('Risk Predictor', 'model', 'risk_prediction'),
    ('T5 Summarizer', 'model', 'summarization'),
    ('Safety Evaluator', 'system', 'evaluation'),
    ('RAG Pipeline', 'system', 'rag'),
    ('FHIR Converter', 'system', 'fhir')
ON CONFLICT DO NOTHING;
