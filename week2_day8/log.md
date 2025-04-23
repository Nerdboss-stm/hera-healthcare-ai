# Week 2 – Day 8 Log

## ✅ Tasks Completed
- Loaded trained model and patient vitals data
- Simulated 10 real-time patient arrivals using `.sample()`
- Ran predictions with `predict()` and `predict_proba()`
- Logged predictions with confidence scores into `prediction_log.txt`

## 🧠 Learnings
- Built simulation of real-time AI inferences
- Modeled prediction loop that mimics real-time healthcare dashboards
- Reinforced understanding of `predict_proba` vs `predict`

## 📁 Files Added
- `live_predict.py`
- `prediction_log.txt`

# Week 2 – Day 8.5 Log: PostgreSQL Logging

## ✅ Completed
- Installed PostgreSQL via Homebrew on macOS
- Created DB `hera_ai`, user `hera`, table `patient_predictions`
- Wrote Python logger using psycopg2
- Inserted predictions with full feature + label info in real-time

## 🧠 Learning
- How to structure real-time logs in a relational DB
- Use parameterized queries to avoid SQL injection
- Commit after each insert or batch insert periodically


