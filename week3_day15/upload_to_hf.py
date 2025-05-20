from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import HfApi, login

# Login to HF Hub
login(token="your_hf_token_here")  # or use CLI `huggingface-cli login`

# Load fine-tuned model
model_path = "./model"  # Adjust if you want to push checkpoint-2700 instead
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Push to HF Hub
model.push_to_hub("clinical-note-summarizer-t5", use_auth_token=True)
tokenizer.push_to_hub("clinical-note-summarizer-t5", use_auth_token=True)

