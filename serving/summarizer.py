from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = "model"  # path to your local fine-tuned model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_summary(note: str, max_length=150, min_length=30) -> str:
    input_ids = tokenizer(note, return_tensors="pt", padding=True, truncation=True).input_ids
    output = model.generate(input_ids, max_length=max_length, min_length=min_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

