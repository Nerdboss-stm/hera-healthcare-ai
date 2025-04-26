def preprocess_function(examples, tokenizer, input_max_length, output_max_length):
    # Prefix input with "summarize:"
    inputs = ["summarize: " + doc for doc in examples["original_note"]]
    
    # Tokenize inputs with padding and truncation
    model_inputs = tokenizer(
        inputs,
        max_length=input_max_length,
        padding="max_length",   # ✅ Added padding
        truncation=True,
    )
    
    # Tokenize labels (target summaries) with padding and truncation
    with tokenizer.as_target_tokenizer():   # ✅ This is safer for T5 tokenizers
        labels = tokenizer(
            examples["target_summary"],
            max_length=output_max_length,
            padding="max_length",   # ✅ Added padding
            truncation=True,
        )

    # Attach labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

