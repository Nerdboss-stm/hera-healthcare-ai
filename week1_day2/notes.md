# Day 2 Notes

# 📘 Week 1 – Day 2: Explore Vitals Dataset

## ✅ Objective
Understand the structure and content of a real-world patient vitals dataset and visualize key time-series signals to build foundational understanding of data trends in healthcare.

---

## 🗂️ Dataset Summary

- **File:** `human_vitals.csv`
- **Rows:** ~1000+ patients
- **Columns Include:**
  - 🫀 `Heart Rate` (bpm)
  - 🫁 `Respiratory Rate`
  - 🌡️ `Body Temperature` (°C)
  - 🧪 `Oxygen Saturation` (%)
  - 🩸 `Systolic/Diastolic Blood Pressure`
  - 📊 Derived: `BMI`, `HRV`, `MAP`, `Pulse Pressure`
  - ⏳ `Timestamp` — when each observation was recorded
  - 🏷️ `Risk Category` — classifies patients into High Risk / Low Risk

---

## 📈 Exploratory Steps

- Parsed and indexed the `Timestamp` column
- Used `matplotlib` to plot time-series line charts for:
  - Heart Rate
  - Respiratory Rate
  - Body Temperature
  - SpO2
- Observed how vitals vary across time

---

## 📊 Plot Summary

- Clear variation observed in heart rate and SpO2 over time
- All 4 vitals plotted on a single timeline to view overlap and correlations
- Saved visualization as: `vital_signs_plot.png`

---

## 🧠 Learnings & Takeaways

- Time-series healthcare data must be indexed properly for meaningful visualization
- Plotting vitals gives an intuitive feel of patient condition trends
- Columns like MAP and Pulse Pressure can be derived from primary BP values
- Sepsis and other deterioration patterns often involve SpO2 drops and HR spikes

---

## 🛠️ Next Steps

- Handle missing values (simulate if needed)
- Add noise to simulate real-world sensor irregularities
- Begin calculating derived metrics manually to verify formulas

---

## 📅 GitHub Activity

- Added: `explore_vitals.py`, `vital_signs_plot.png`, `human_vitals.csv`, `notes.md`
- Commit message: `Loaded human vitals dataset, visualized key vital signs`
