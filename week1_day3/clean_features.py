import pandas as pd
import numpy as np

df = pd.read_csv("../week1_day2/human_vitals.csv")

# Show null counts
print("Missing values before:\n", df.isnull().sum())

# Strategy: fill numerical columns with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Optional: fill categorical values with mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

print("Missing values after:\n", df.isnull().sum())

# Add slight random noise to vitals
for col in ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation']:
    noise = np.random.normal(0, 0.5, size=len(df))
    df[col + '_Noisy'] = df[col] + noise
# BMI = weight (kg) / height (m)^2
df['Calculated_BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
print("Calculated_BMI",df['Calculated_BMI'])

# MAP = (2 * diastolic + systolic) / 3
df['Calculated_MAP'] = (2 * df['Diastolic Blood Pressure'] + df['Systolic Blood Pressure']) / 3
print("Calculated_MAP",df['Calculated_MAP'])

# Pulse Pressure = systolic - diastolic
df['Calculated_Pulse_Pressure'] = df['Systolic Blood Pressure'] - df['Diastolic Blood Pressure']
print("Calculated_Pulse_Pressure",df['Calculated_Pulse_Pressure'])


df.to_csv("cleaned_vitals.csv", index=False)
