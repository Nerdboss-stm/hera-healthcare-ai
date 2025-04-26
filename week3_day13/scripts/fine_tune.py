import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print("\n=== sys.path BEFORE ANYTHING ===")
for p in sys.path:
    print(p)
print("===============================\n")
# Dynamically add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)  # Insert at index 0 (highest priority)

print("\n=== sys.path AFTER PATCH ===")
for p in sys.path:
    print(p)
print("============================\n")
import yaml
import pandas as pd
import torch
from datasets import DatasetDict, Dataset
from transformers import Seq2SeqTrainingArguments
from src.model import load_model_and_tokenizer
from src.dataset import preprocess_function
from src.trainer import build_trainer
from src.trainer import build_compute_metrics

# FORCE PyTorch device to CPU manually
device = torch.device('cpu')

# Load config
with open("week3_day13/configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data
data = pd.read_csv("week3_day12/notes_1000.csv")
train_size = int(0.9 * len(data))
train_dataset = Dataset.from_pandas(data.iloc[:train_size])
val_dataset = Dataset.from_pandas(data.iloc[train_size:])
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Load model
model, tokenizer = load_model_and_tokenizer(config["model_checkpoint"])

# Preprocess
tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer, config["input_max_length"], config["output_max_length"]), batched=True)

# Training Args
training_args = Seq2SeqTrainingArguments(
    output_dir="./model",
    eval_strategy="steps",
    learning_rate=float(config["learning_rate"]),
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=config["weight_decay"],
    save_total_limit=2,
    save_steps=config["save_steps"],
    eval_steps=config["eval_steps"],
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,
    no_cuda=True
)

# Build Trainer
trainer = build_trainer(model, tokenizer, tokenized_datasets, training_args)

# Train
trainer.train()

# Save
trainer.save_model("./model")
print("✅ Fine-tuned model saved!")

