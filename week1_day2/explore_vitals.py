import pandas as pd
import matplotlib.pyplot as plt

# Load the vitals CSV
df = pd.read_csv('human_vitals.csv')

# Show first few rows
print("First 5 rows:")
print(df.head())

# Print column info
print("\nColumn Summary:")
print(df.info())

# Plot temperature trend
df['Temp'].plot(title='Simulated Vital Trend - Temperature', figsize=(10, 4))
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.tight_layout()
plt.savefig("hera-healthcare-ai/week1_day2/vitals_plot.png")
plt.show()
