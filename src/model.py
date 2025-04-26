from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model_and_tokenizer(model_checkpoint):
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    return model, tokenizer

