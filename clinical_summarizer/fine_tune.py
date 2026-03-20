import sys
import os

import yaml  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import DatasetDict, Dataset  # noqa: E402
from transformers import Seq2SeqTrainingArguments  # noqa: E402

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_dir, ".."))
sys.path.insert(0, project_root)

from clinical_summarizer.model import load_model_and_tokenizer  # noqa: E402
from clinical_summarizer.dataset import preprocess_function  # noqa: E402
from clinical_summarizer.trainer import build_trainer  # noqa: E402

# FORCE PyTorch device to CPU manually
device = torch.device("cpu")

# Load config
with open(os.path.join(_dir, "configs", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Load data
data = pd.read_csv(os.path.join(_dir, "..", "data", "clinical_notes", "notes_1000.csv"))
train_size = int(0.9 * len(data))
train_dataset = Dataset.from_pandas(data.iloc[:train_size])
val_dataset = Dataset.from_pandas(data.iloc[train_size:])
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Load model
model, tokenizer = load_model_and_tokenizer(config["model_checkpoint"])

# Preprocess
tokenized_datasets = dataset.map(
    lambda x: preprocess_function(
        x, tokenizer, config["input_max_length"], config["output_max_length"]
    ),
    batched=True,
)

model_output_dir = os.path.join(project_root, "model")

# Training Args
training_args = Seq2SeqTrainingArguments(
    output_dir=model_output_dir,
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
    logging_dir=os.path.join(project_root, "logs"),
    logging_steps=10,
    fp16=False,
    no_cuda=True,
)

# Build Trainer
trainer = build_trainer(model, tokenizer, tokenized_datasets, training_args)

# Train
trainer.train()

# Save
trainer.save_model(model_output_dir)
print("Fine-tuned model saved to:", model_output_dir)
