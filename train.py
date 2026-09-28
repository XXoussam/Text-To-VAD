"""Fine-tune DeBERTa-v3 to predict Valence / Arousal / Dominance on EmoBank.

    Input text → DeBERTa-v3-base encoder → pooled [CLS] (768) → Linear(768 → 3) → [V, A, D]

Example:
    python train.py --epochs 5 --output-dir ./deberta_vad
"""

import argparse
import dataclasses
import json
import os

import text_to_vad  # noqa: F401  -- first: switches HTTPS to the OS certificate store

import pandas as pd
import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from text_to_vad.config import EMOBANK_URL, LABEL_COLS, MAX_LENGTH, MODEL_NAME, OUTPUT_DIR, SEED, from_unit
from text_to_vad.data import build_datasets, load_emobank, tokenize_datasets
from text_to_vad.metrics import compute_metrics, plot_predictions


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model id to fine-tune")
    p.add_argument("--data", default=EMOBANK_URL, help="EmoBank CSV (URL or local path)")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2, help="Early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-fp16", action="store_true", help="Disable mixed precision on GPU")
    return p.parse_args()


def warmup_kwargs(ratio: float) -> dict:
    """transformers 5 dropped `warmup_ratio`; `warmup_steps` accepts a float in [0, 1) as a ratio instead."""
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    return {"warmup_ratio": ratio} if "warmup_ratio" in fields else {"warmup_steps": ratio}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("transformers:", transformers.__version__)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (training on CPU will be slow)")

    # Data
    df = load_emobank(args.data)
    print(df.shape)
    print(df["split"].value_counts())
    print(df[["V", "A", "D"]].describe())

    raw_ds = build_datasets(df)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_ds = tokenize_datasets(raw_ds, tokenizer, args.max_length)
    print(tokenized_ds)

    # Model: 3-output linear head trained with MSE loss. The warning about
    # newly initialized `classifier` / `pooler` weights is expected.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_COLS),
        problem_type="regression",
    )
    model.config.id2label = {i: n for i, n in enumerate(LABEL_COLS)}
    model.config.label2id = {n: i for i, n in enumerate(LABEL_COLS)}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        **warmup_kwargs(args.warmup_ratio),
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="pearson_mean",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    # Held-out test set
    pred_out = trainer.predict(tokenized_ds["test"], metric_key_prefix="test")
    test_metrics = pred_out.metrics
    print(pd.Series(test_metrics).round(4).to_string())
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    plot_path = os.path.join(args.output_dir, "test_scatter.png")
    plot_predictions(from_unit(pred_out.predictions), from_unit(pred_out.label_ids), plot_path)
    print(f"Saved plot to {plot_path}")

    # Save best model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
