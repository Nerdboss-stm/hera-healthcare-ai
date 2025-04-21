import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("../week1_day3/cleaned_vitals.csv")
df['Risk Category'] = df['Risk Category'].map({'High Risk': 1, 'Low Risk': 0})

features = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
            'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 
            'Calculated_BMI', 'Calculated_MAP']
X = df[features]
y = df['Risk Category']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate on test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(best_model, "tuned_risk_model.pkl")

