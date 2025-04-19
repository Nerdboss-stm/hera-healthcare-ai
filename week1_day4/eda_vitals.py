import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../week1_day3/cleaned_vitals.csv")

# Distribution of Heart Rate
sns.histplot(df['Heart Rate'], kde=True)
plt.title("Heart Rate Distribution")
plt.savefig("heart_rate_distribution.png")
plt.clf()

# Compare risk categories with box plots
sns.boxplot(x='Risk Category', y='Heart Rate', data=df)
plt.title("Heart Rate by Risk Category")
plt.savefig("hr_by_risk.png")
plt.clf()

# BMI vs Age
sns.scatterplot(x='Age', y='Calculated_BMI', hue='Risk Category', data=df)
plt.title("BMI vs Age Colored by Risk")
plt.savefig("bmi_age_risk.png")
plt.clf()

# Correlation matrix
correlation = df.select_dtypes(include='number').corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap_correlation.png")

