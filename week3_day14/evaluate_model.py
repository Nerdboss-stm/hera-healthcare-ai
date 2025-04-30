import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import evaluate
import torch
from tqdm import tqdm

# ========== CONFIG ==========
MODEL_DIR = "../model"
DATA_FILE = "../week3_day12/notes_1000.csv"
OUTPUT_METRICS_FILE = "evaluation_report.md"

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_FILE)
print("Columns in DataFrame:", df.columns.tolist())
print(df.head(1))
df = df.rename(columns={"text": "input_text", "summary": "target_text"})
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ========== LOAD MODEL ==========
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ========== GENERATE SUMMARIES ==========
predictions = []
references = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    input_ids = tokenizer(row['original_note'], return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids.to(device)
    outputs = model.generate(input_ids, max_length=128, num_beams=4)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred)
    references.append(row['target_summary'])

# ========== EVALUATE ==========
rouge = evaluate.load("rouge")
rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

try:
    bertscore = evaluate.load("bertscore")
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    bert_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
except:
    bert_f1 = "BERTScore evaluation failed (likely not installed)."

# ========== SAVE METRICS ==========
with open(OUTPUT_METRICS_FILE, "w") as f:
    f.write("# Evaluation Report\n\n")
    f.write("## ROUGE Scores\n")
    for k, v in rouge_scores.items():
        f.write(f"- **{k}**: {v:.4f}\n")
    f.write("\n## BERTScore\n")
    f.write(f"- **BERTScore F1**: {bert_f1}\n")

print("✅ Evaluation complete. Metrics saved to:", OUTPUT_METRICS_FILE)

