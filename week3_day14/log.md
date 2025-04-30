✅ Tasks Completed

Loaded fine-tuned T5-small model and tokenizer from week3_day13/model/

Split notes_1000.csv into 80% training, 20% testing using train_test_split

Wrote evaluate_model.py script:

Loads model, tokenizer, test data

Generates summaries with generate()

Evaluates results using ROUGE and BERTScore

Saves output to evaluation_report.md

Configured .gitignore to prevent pushing large models/logs

📊 Metrics Computed

ROUGE-1, ROUGE-2, ROUGE-L (with stemming)

BERTScore F1 (if package installed)

🛠️ Issues Encountered

Handled tokenizer decoding and GPU/CPU switching

Avoided pushing .bin files to GitHub using .gitignore

Managed evaluation in low-resource environment (optional batch size reduction needed)


