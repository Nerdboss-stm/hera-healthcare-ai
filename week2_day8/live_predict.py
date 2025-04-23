import pandas as pd
import joblib
import time
import psycopg2
from database.db_config import db_config  # import credentials

# Step 1: Connect to PostgreSQL
try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL database.")
except Exception as e:
    print("❌ Failed to connect to DB:", e)
    exit()

# Step 2: Load model and data
model = joblib.load("../week2_day6/tuned_risk_model.pkl")
df = pd.read_csv("../week1_day3/cleaned_vitals.csv")

features = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age',
    'Calculated_BMI', 'Calculated_MAP'
]

label_map = {1: "High Risk", 0: "Low Risk"}

# Step 3: Simulate real-time predictions
for i in range(10):
    patient = df.sample(1)
    X = patient[features]
    y_pred = model.predict(X)[0]
    y_prob = model.predict_proba(X)[0][y_pred]
    label = label_map[y_pred]

    # Prepare values for insertion
    values = (
        float(patient['Heart Rate']),
        float(patient['Respiratory Rate']),
        float(patient['Body Temperature']),
        float(patient['Oxygen Saturation']),
        float(patient['Systolic Blood Pressure']),
        float(patient['Diastolic Blood Pressure']),
        int(patient['Age']),
        float(patient['Calculated_BMI']),
        float(patient['Calculated_MAP']),
        label,
        round(y_prob, 4)
    )

    # Insert into DB
    insert_query = """
        INSERT INTO patient_predictions (
            heart_rate, resp_rate, temperature, spo2,
            systolic_bp, diastolic_bp, age, bmi, map,
            predicted_label, confidence
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    try:
        cursor.execute(insert_query, values)
        conn.commit()
        print(f"🧬 Patient {i+1}: {label} (Confidence: {y_prob:.2f}) → Logged.")
    except Exception as e:
        print(f"❌ Error logging Patient {i+1}:", e)

    time.sleep(1)

# Step 4: Cleanup
cursor.close()
conn.close()
print("✅ All predictions logged and DB connection closed.")

