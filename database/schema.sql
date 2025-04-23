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

