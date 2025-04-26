import evaluate
from transformers import Trainer

rouge = evaluate.load("rouge")

import numpy as np

def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.argmax(preds, axis=-1) if preds.ndim == 3 else preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}

        prediction_lens = [len(pred.split()) for pred in decoded_preds]
        result["gen_len"] = sum(prediction_lens) / len(prediction_lens)

        return result
    return compute_metrics

def build_trainer(model, tokenizer, tokenized_datasets, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer)  # ✅ Correct!
    )
    return trainer

