import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load cleaned dataset
df = pd.read_csv("../week1_day3/cleaned_vitals.csv")

# Encode categorical label
df['Risk Category'] = df['Risk Category'].map({'High Risk': 1, 'Low Risk': 0})

# Define features & target
features = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
            'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'Calculated_BMI', 'Calculated_MAP']
X = df[features]
y = df['Risk Category']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "risk_predictor_model.pkl")

