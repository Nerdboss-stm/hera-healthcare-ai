---
language: en
license: mit
tags:
- summarization
- clinical-notes
- healthcare
- t5
- transformers
- explainable-ai
datasets:
- custom
metrics:
- rouge
- bertscore
---

# 🏥 Clinical Note Summarizer (Fine-tuned T5)

## 📘 Model Description

This model is a fine-tuned version of T5-Base for the task of abstractive summarization of clinical notes. It is part of a full-stack real-time AI healthcare pipeline that supports summarization, explainability (SHAP), alerting, and API deployment.

## 📊 Dataset

Custom-generated synthetic clinical notes with advanced vital signs metadata. Used `generate_realistic_notes_advanced.py` for realism.

## ⚙️ Training Details

- Base model: `t5-base`
- Epochs: 5
- Batch size: 16
- Evaluation metrics: ROUGE, BERTScore
- Fine-tuned using Hugging Face `Trainer`

## 📈 Metrics

| Metric     | Score   |
|------------|---------|
| ROUGE-L    | 0.44    |
| BERTScore  | 0.89    |

## 🔍 Explainability

- SHAP used to interpret features used during training.
- SHAP plots available in `week2_day7/shap_summary.png`.

## ✅ Intended Use

- Summarization of clinical notes in real-time.
- Designed to assist doctors, researchers, and EHR providers.

## ⚠️ Limitations

- Not for diagnostic purposes.
- Bias may exist due to synthetic note generation.

## 🧑‍💻 Author

Saran Teja Mallela ([@Nerdboss-stm](https://github.com/Nerdboss-stm))

---


