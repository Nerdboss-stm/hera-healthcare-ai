from flask import Flask
from flask_cors import CORS
from prometheus_client import start_http_server, Gauge
import pandas as pd
import joblib
import socket

app = Flask(__name__)
CORS(app)

# Load model and data
model = joblib.load("../week2_day6/tuned_risk_model.pkl")
df = pd.read_csv("../week1_day3/cleaned_vitals.csv")
features = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age',
    'Calculated_BMI', 'Calculated_MAP'
]

# Metrics
total_predictions = Gauge('total_predictions', 'Total predictions made')
high_risk_predictions = Gauge('high_risk_predictions', 'High risk patients predicted')
last_prediction = Gauge('last_prediction', 'Last prediction (0=Low, 1=High)')

@app.route("/predict", methods=['GET'], strict_slashes=False)
def predict():
    patient = df.sample(1)[features]
    pred = model.predict(patient)[0]

    total_predictions.inc()
    last_prediction.set(pred)
    if pred == 1:
        high_risk_predictions.inc()

    return {"prediction": int(pred)}

if __name__ == '__main__':
    try:
        start_http_server(9100)  # Changed from 8000 to 9100 to avoid port conflicts
    except OSError:
        print("⚠️ Port 9100 already in use. Please check or restart.")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

