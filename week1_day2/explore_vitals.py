import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("human_vitals.csv")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set Timestamp as index for time-series plotting
df.set_index('Timestamp', inplace=True)

# Plot multiple vitals
plt.figure(figsize=(12, 6))
plt.plot(df['Heart Rate'], label='Heart Rate (BPM)')
plt.plot(df['Respiratory Rate'], label='Respiratory Rate')
plt.plot(df['Body Temperature'], label='Body Temp (°C)')
plt.plot(df['Oxygen Saturation'], label='SpO2 (%)')
plt.title('Vital Signs Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vital_signs_plot(1).png")
plt.show()
