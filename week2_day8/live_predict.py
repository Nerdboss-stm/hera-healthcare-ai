import pandas as pd
import joblib
import time

# Load cleaned data
df = pd.read_csv("../week1_day3/cleaned_vitals.csv")

# Encode target (optional, for display)
label_map = {1: "High Risk", 0: "Low Risk"}
features = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age',
    'Calculated_BMI', 'Calculated_MAP'
]

# Load trained model
model = joblib.load("../week2_day6/tuned_risk_model.pkl")

# Open a log file
log_file = open("prediction_log.txt", "w")
log_file.write("Live Patient Risk Predictions\n")
log_file.write("="*35 + "\n\n")

# Simulate 10 patients arriving one by one
for i in range(10):
    patient = df.sample(1)[features]
    pred = model.predict(patient)[0]
    prob = model.predict_proba(patient)[0][pred]
    label = label_map[pred]

    log = f"👤 Patient {i+1}: Predicted = {label} (Confidence: {prob:.2f})"
    print(log)
    log_file.write(log + "\n")

    time.sleep(1)  # Simulate delay (optional)

log_file.close()
print("\n✅ Predictions saved to prediction_log.txt")

