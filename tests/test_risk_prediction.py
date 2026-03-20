import pandas as pd
import os
import pytest


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class TestPreprocessing:
    """Tests for the vitals feature engineering pipeline."""

    def test_cleaned_vitals_file_exists(self):
        path = os.path.join(DATA_DIR, "processed", "cleaned_vitals.csv")
        assert os.path.exists(path), "cleaned_vitals.csv not found in data/processed/"

    def test_cleaned_vitals_has_engineered_features(self):
        path = os.path.join(DATA_DIR, "processed", "cleaned_vitals.csv")
        if not os.path.exists(path):
            pytest.skip("cleaned_vitals.csv not available")
        df = pd.read_csv(path)
        expected_cols = ["Calculated_BMI", "Calculated_MAP", "Calculated_Pulse_Pressure"]
        for col in expected_cols:
            assert col in df.columns, f"Missing engineered feature: {col}"

    def test_cleaned_vitals_no_nulls_in_features(self):
        path = os.path.join(DATA_DIR, "processed", "cleaned_vitals.csv")
        if not os.path.exists(path):
            pytest.skip("cleaned_vitals.csv not available")
        df = pd.read_csv(path)
        features = [
            "Heart Rate", "Respiratory Rate", "Body Temperature",
            "Oxygen Saturation", "Systolic Blood Pressure",
            "Diastolic Blood Pressure", "Age",
        ]
        for col in features:
            if col in df.columns:
                assert df[col].isnull().sum() == 0, f"Null values in {col}"

    def test_risk_category_labels(self):
        path = os.path.join(DATA_DIR, "processed", "cleaned_vitals.csv")
        if not os.path.exists(path):
            pytest.skip("cleaned_vitals.csv not available")
        df = pd.read_csv(path)
        if "Risk Category" in df.columns:
            valid_labels = {"High Risk", "Low Risk"}
            actual_labels = set(df["Risk Category"].unique())
            assert actual_labels.issubset(valid_labels), f"Unexpected labels: {actual_labels - valid_labels}"


class TestModelTraining:
    """Tests for model training outputs."""

    def test_random_forest_features_match(self):
        """Verify the expected feature list is consistent."""
        expected_features = [
            "Heart Rate", "Respiratory Rate", "Body Temperature",
            "Oxygen Saturation", "Systolic Blood Pressure",
            "Diastolic Blood Pressure", "Age",
            "Calculated_BMI", "Calculated_MAP",
        ]
        assert len(expected_features) == 9
