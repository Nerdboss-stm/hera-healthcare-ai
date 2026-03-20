"""Exploratory Data Analysis for patient vitals dataset."""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(_dir, "..", "data", "processed", "cleaned_vitals.csv")

df = pd.read_csv(data_path)

# Distribution of Heart Rate
sns.histplot(df['Heart Rate'], kde=True)
plt.title("Heart Rate Distribution")
plt.savefig(os.path.join(_dir, "heart_rate_distribution.png"))
plt.clf()

# Compare risk categories with box plots
sns.boxplot(x='Risk Category', y='Heart Rate', data=df)
plt.title("Heart Rate by Risk Category")
plt.savefig(os.path.join(_dir, "hr_by_risk.png"))
plt.clf()

# BMI vs Age
sns.scatterplot(x='Age', y='Calculated_BMI', hue='Risk Category', data=df)
plt.title("BMI vs Age Colored by Risk")
plt.savefig(os.path.join(_dir, "bmi_age_risk.png"))
plt.clf()

# Correlation matrix
correlation = df.select_dtypes(include='number').corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(_dir, "heatmap_correlation.png"))
