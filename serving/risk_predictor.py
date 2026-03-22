import os
import numpy as np
import joblib

_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_dir, "..", "risk_prediction", "tuned_risk_model.pkl")
FALLBACK_PATH = os.path.join(_dir, "..", "risk_prediction", "risk_predictor_model.pkl")

_model = None

FEATURES = [
    "Heart Rate",
    "Respiratory Rate",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Age",
    "Calculated_BMI",
    "Calculated_MAP",
]

LABEL_MAP = {1: "High Risk", 0: "Low Risk"}

# 4-level risk thresholds based on clinical severity scoring
RISK_THRESHOLDS = [
    (0.75, "Critical"),  # risk_score >= 0.75
    (0.50, "High"),      # risk_score >= 0.50
    (0.25, "Medium"),    # risk_score >= 0.25
    (0.00, "Low"),       # risk_score < 0.25
]


def _classify_risk_level(risk_score: float) -> str:
    """Map continuous risk_score (0-1) to 4-level label."""
    for threshold, label in RISK_THRESHOLDS:
        if risk_score >= threshold:
            return label
    return "Low"


def _compute_clinical_risk_score(
    heart_rate: float,
    respiratory_rate: float,
    body_temperature: float,
    oxygen_saturation: float,
    systolic_bp: float,
    diastolic_bp: float,
    age: int,
    ml_risk_prob: float,
) -> float:
    """Blend ML probability with clinical heuristics for a nuanced risk score.

    The binary ML model clusters at extremes (0.05 or 0.95). This blends in
    a clinical severity component so that moderate cases land in the middle.
    """
    score = 0.0

    # Vital sign deviations — wider tiers for smoother distribution
    # Heart rate
    if heart_rate < 40 or heart_rate > 150:
        score += 0.15
    elif heart_rate < 50 or heart_rate > 130:
        score += 0.10
    elif heart_rate < 60 or heart_rate > 110:
        score += 0.06
    elif heart_rate < 65 or heart_rate > 100:
        score += 0.03

    # Respiratory rate
    if respiratory_rate < 6 or respiratory_rate > 35:
        score += 0.15
    elif respiratory_rate < 8 or respiratory_rate > 30:
        score += 0.10
    elif respiratory_rate < 12 or respiratory_rate > 22:
        score += 0.06
    elif respiratory_rate < 14 or respiratory_rate > 20:
        score += 0.03

    # Oxygen saturation
    if oxygen_saturation < 85:
        score += 0.18
    elif oxygen_saturation < 88:
        score += 0.12
    elif oxygen_saturation < 92:
        score += 0.08
    elif oxygen_saturation < 95:
        score += 0.04
    elif oxygen_saturation < 97:
        score += 0.02

    # Blood pressure
    if systolic_bp < 80 or systolic_bp > 200:
        score += 0.15
    elif systolic_bp < 85 or systolic_bp > 180:
        score += 0.10
    elif systolic_bp < 100 or systolic_bp > 160:
        score += 0.06
    elif systolic_bp < 110 or systolic_bp > 140:
        score += 0.03

    # Temperature
    if body_temperature < 35.0 or body_temperature > 40.0:
        score += 0.12
    elif body_temperature < 35.5 or body_temperature > 39.5:
        score += 0.08
    elif body_temperature < 36.0 or body_temperature > 38.5:
        score += 0.05
    elif body_temperature < 36.3 or body_temperature > 37.8:
        score += 0.02

    # Diastolic BP
    if diastolic_bp < 40 or diastolic_bp > 120:
        score += 0.10
    elif diastolic_bp < 50 or diastolic_bp > 100:
        score += 0.06
    elif diastolic_bp < 60 or diastolic_bp > 90:
        score += 0.03

    # Age factor — more granular
    if age > 85:
        score += 0.12
    elif age > 75:
        score += 0.08
    elif age > 65:
        score += 0.05
    elif age > 55:
        score += 0.03
    elif age < 2:
        score += 0.08
    elif age < 5:
        score += 0.06
    elif age < 12:
        score += 0.03

    # Blend: 35% ML probability + 65% clinical heuristic (capped at 1.0)
    # This ensures clinical signals dominate, producing all 4 risk levels
    clinical_score = min(score, 0.85)
    blended = 0.35 * ml_risk_prob + 0.65 * clinical_score
    return round(min(blended, 1.0), 4)


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
    calculated_bmi = bmi_weight / (bmi_height**2)
    calculated_map = (2 * diastolic_bp + systolic_bp) / 3

    features = np.array(
        [
            [
                heart_rate,
                respiratory_rate,
                body_temperature,
                oxygen_saturation,
                systolic_bp,
                diastolic_bp,
                age,
                calculated_bmi,
                calculated_map,
            ]
        ]
    )

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    ml_prob = float(probabilities[1])

    # Compute blended clinical risk score for 4-level classification
    risk_score = _compute_clinical_risk_score(
        heart_rate, respiratory_rate, body_temperature,
        oxygen_saturation, systolic_bp, diastolic_bp,
        age, ml_prob,
    )
    risk_level = _classify_risk_level(risk_score)

    return {
        "prediction": risk_level,
        "confidence": round(float(probabilities[prediction]), 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "ml_binary_label": LABEL_MAP[prediction],
        "ml_probability": round(ml_prob, 4),
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
