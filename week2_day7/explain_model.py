import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Step 1: Load data
df = pd.read_csv("../week1_day3/cleaned_vitals.csv")
df['Risk Category'] = df['Risk Category'].map({'High Risk': 1, 'Low Risk': 0})

features = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age',
    'Calculated_BMI', 'Calculated_MAP'
]

X = pd.DataFrame(df[features].copy(), columns=features)

# Step 2: Load trained model
model = joblib.load("../week2_day6/tuned_risk_model.pkl")

# Step 3: Wrap the model
class ModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def predict(self, X):
        return self.model.predict(X)

wrapped_model = ModelWrapper(model)

# Step 4: SHAP explainer on a sample
X_sample = X.sample(n=500, random_state=42)
explainer = shap.Explainer(wrapped_model.predict_proba, X_sample)

shap_values = explainer(X_sample)
print("✅ SHAP values shape:", shap_values.values.shape)
print("✅ Sample data shape:", X_sample.shape)

# Step 5: Beeswarm plot with safe fallback to matplotlib
plt.figure()
shap.plots.beeswarm(shap_values[:, :, 1], show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
print("✅ SHAP summary plot saved as 'shap_summary.png'")

