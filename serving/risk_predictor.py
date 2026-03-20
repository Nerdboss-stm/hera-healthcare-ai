import os
import numpy as np
import joblib

_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_dir, "..", "risk_prediction", "tuned_risk_model.pkl")
FALLBACK_PATH = os.path.join(_dir, "..", "risk_prediction", "risk_predictor_model.pkl")

_model = None

FEATURES = [
    "Heart Rate", "Respiratory Rate", "Body Temperature", "Oxygen Saturation",
    "Systolic Blood Pressure", "Diastolic Blood Pressure", "Age",
    "Calculated_BMI", "Calculated_MAP",
]

LABEL_MAP = {1: "High Risk", 0: "Low Risk"}


def _load_risk_model():
    global _model
    if _model is None:
        for path in [MODEL_PATH, FALLBACK_PATH]:
            if os.path.exists(path):
                _model = joblib.load(path)
                return _model
        raise FileNotFoundError(
            "Risk prediction model not found. "
            "Run 'python -m risk_prediction.train' then 'python -m risk_prediction.tune' first."
        )
    return _model


def predict_risk(
    heart_rate: float,
    respiratory_rate: float,
    body_temperature: float,
    oxygen_saturation: float,
    systolic_bp: float,
    diastolic_bp: float,
    age: int,
) -> dict:
    model = _load_risk_model()

    bmi_weight = 70.0
    bmi_height = 1.70
    calculated_bmi = bmi_weight / (bmi_height ** 2)
    calculated_map = (2 * diastolic_bp + systolic_bp) / 3

    features = np.array([[
        heart_rate, respiratory_rate, body_temperature, oxygen_saturation,
        systolic_bp, diastolic_bp, age, calculated_bmi, calculated_map,
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction])
    label = LABEL_MAP[prediction]

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "risk_score": round(float(probabilities[1]), 4),
        "features_used": {
            "heart_rate": heart_rate,
            "respiratory_rate": respiratory_rate,
            "body_temperature": body_temperature,
            "oxygen_saturation": oxygen_saturation,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "age": age,
            "calculated_bmi": round(calculated_bmi, 2),
            "calculated_map": round(calculated_map, 2),
        },
    }
