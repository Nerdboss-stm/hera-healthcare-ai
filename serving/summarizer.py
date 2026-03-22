import os
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)

_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_dir, "..", "model")
FALLBACK_MODEL = "t5-small"

_model = None
_tokenizer = None
_using_fallback = False


def _load_model():
    global _model, _tokenizer, _using_fallback
    if _model is None:
        # Try fine-tuned model first
        if os.path.exists(MODEL_PATH):
            try:
                _tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
                _model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
                # Validate: check if model produces non-empty output
                test_ids = _tokenizer(
                    "summarize: Patient has chest pain.",
                    return_tensors="pt",
                    truncation=True,
                ).input_ids
                test_out = _model.generate(test_ids, max_length=50)
                test_text = _tokenizer.decode(test_out[0], skip_special_tokens=True)
                if test_text.strip():
                    logger.info("Fine-tuned model loaded and validated")
                    return _model, _tokenizer
                else:
                    logger.warning(
                        "Fine-tuned model produces empty output, falling back to %s",
                        FALLBACK_MODEL,
                    )
                    _model, _tokenizer = None, None
            except Exception as e:
                logger.warning("Failed to load fine-tuned model: %s", e)
                _model, _tokenizer = None, None

        # Fallback to pretrained t5-small
        logger.info("Loading fallback model: %s", FALLBACK_MODEL)
        _tokenizer = T5Tokenizer.from_pretrained(FALLBACK_MODEL)
        _model = T5ForConditionalGeneration.from_pretrained(FALLBACK_MODEL)
        _using_fallback = True
    return _model, _tokenizer


def generate_summary(note: str, max_length=150, min_length=30) -> str:
    model, tokenizer = _load_model()
    # T5 requires a task prefix for summarization
    prefix = "summarize: " if _using_fallback else ""
    input_text = f"{prefix}{note}"
    input_ids = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).input_ids
    output = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
