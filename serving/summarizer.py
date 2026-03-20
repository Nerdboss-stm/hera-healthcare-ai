import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_dir, "..", "model")

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run 'python -m clinical_summarizer.fine_tune' to train, "
                "or download the fine-tuned model."
            )
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        _model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    return _model, _tokenizer


def generate_summary(note: str, max_length=150, min_length=30) -> str:
    model, tokenizer = _load_model()
    input_ids = tokenizer(note, return_tensors="pt", padding=True, truncation=True).input_ids
    output = model.generate(input_ids, max_length=max_length, min_length=min_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)
