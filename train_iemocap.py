"""Fine-tune the VAD model on IEMOCAP transcripts (spoken conversation).

The EmoBank model (./deberta_vad) was trained on written text whose scores
cluster near neutral, so on IEMOCAP its predictions are squeezed toward 3
(prediction spread 0.2-0.4 vs human 0.7-0.9). This script continues training
it on IEMOCAP's real transcripts and human V/A/D scores.

Split (speaker-independent, the usual IEMOCAP protocol):
    test        --test-sessions (default Ses05): actors never seen in training
    validation  --val-fraction of the remaining *dialogues* (whole conversations,
                so lines of one scene are never in both train and validation)
    train       the rest of Ses01-Ses04

Loss: 1 - CCC (concordance correlation) by default. MSE rewards predicting
near the mean, which is how the EmoBank model ended up squeezed; CCC also
penalises a wrong mean and a too-narrow spread. --loss mse or ccc+mse to compare.

Example:
    python train_iemocap.py                                   # EmoBank model -> deberta_vad_iemocap
    python train_iemocap.py --init-model microsoft/deberta-v3-base --output-dir ./deberta_vad_iemocap_base
    python train_iemocap.py --text-source asr --output-dir ./deberta_vad_iemocap_asr     # train on Parakeet's text
    python train_iemocap.py --text-source both --output-dir ./deberta_vad_iemocap_both   # real + Parakeet text

Then evaluate the full pipeline on the held-out session, from Parakeet-to-VAD:
    ..\\FineTune\\asr-env\\Scripts\\python evaluate_iemocap.py --sessions Ses05 \\
        --vad-model ..\\Text-To-VAD\\deberta_vad_iemocap --out-dir output\\iemocap_ses05_finetuned
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import text_to_vad  # noqa: F401  -- first: switches HTTPS to the OS certificate store

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from text_to_vad.config import LABEL_COLS, MAX_LENGTH, OUTPUT_DIR, SEED, from_unit, to_unit
from text_to_vad.inference import _no_mistral_regex_patch
from text_to_vad.metrics import plot_predictions, regression_metrics
from train import plot_residuals, plot_training_curves, warmup_kwargs

IEMOCAP_DIR = Path(__file__).resolve().parent.parent / "data" / "IEMOCAP"
# Base-Parakeet transcripts of all 10,039 IEMOCAP clips, written by Parakeet-to-VAD/evaluate_iemocap.py
ASR_CACHE = (Path(__file__).resolve().parent.parent / "Parakeet-to-VAD" / "output" / "iemocap_ses05_finetuned"
             / "asr_cache.jsonl")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iemocap-dir", type=Path, default=IEMOCAP_DIR, help="Folder with metadata.csv")
    p.add_argument("--init-model", default=OUTPUT_DIR,
                   help="Model to start from: the EmoBank model (default) or a Hugging Face id")
    p.add_argument("--output-dir", default="./deberta_vad_iemocap")
    p.add_argument("--test-sessions", nargs="+", default=["Ses05"])
    p.add_argument("--val-fraction", type=float, default=0.1, help="Share of training dialogues used for validation")
    p.add_argument("--loss", choices=["ccc", "mse", "ccc+mse"], default="ccc")
    p.add_argument("--text-source", choices=["ref", "asr", "both"], default="ref",
                   help="Training text: real transcripts (ref), Parakeet's (asr), or both (see split())")
    p.add_argument("--asr-cache", type=Path, default=ASR_CACHE,
                   help="Parakeet transcripts (asr_cache.jsonl from Parakeet-to-VAD), for --text-source asr/both")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32, help="CCC is computed per batch: keep it >= 16")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=float, default=8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2, help="Early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-fp16", action="store_true", help="Disable mixed precision on GPU")
    return p.parse_args()


# ------------------------------------------------------------------ data

def dialogue_id(file: str) -> str:
    """Ses01F_impro01_F000.wav -> Ses01F_impro01 (the conversation the line belongs to)."""
    return re.sub(r"_[FM]\d+$", "", Path(file).stem)


def load_iemocap(iemocap_dir: Path) -> pd.DataFrame:
    path = iemocap_dir / "metadata.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run data/IEMOCAP/import_iemocap.py first, or pass --iemocap-dir")
    df = pd.read_csv(path)
    df["text"] = df["transcription"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["dialogue"] = df["file"].map(dialogue_id)
    for col in LABEL_COLS:  # 1-5 -> 0-1, same scaling as EmoBank training
        df[col] = to_unit(df[col].astype(float))
    return df


def load_asr_texts(cache_path: Path) -> dict[str, str]:
    """file -> Parakeet transcript, from Parakeet-to-VAD's asr_cache.jsonl."""
    texts = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                texts[entry["file"]] = (entry.get("text") or "").strip()
    return texts


def with_text(part: pd.DataFrame, asr_texts: dict[str, str]) -> pd.DataFrame:
    """The same clips with Parakeet's transcript as text; clips Parakeet left empty are dropped."""
    part = part.copy()
    part["text"] = part["file"].map(asr_texts).fillna("")
    return part[part["text"].str.len() > 0]


def split(df: pd.DataFrame, test_sessions: list[str], val_fraction: float, seed: int,
          text_source: str = "ref", asr_texts: dict[str, str] | None = None) -> DatasetDict:
    """The split itself never depends on text_source, so all variants use the same clips.

    text_source: ref  = real transcripts everywhere (default)
                 asr  = train and validation on Parakeet's transcripts (what the model sees in the pipeline)
                 both = training set twice (real + Parakeet transcripts), validation on Parakeet's
    The test set keeps the real transcripts here; evaluate the pipeline with evaluate_iemocap.py.
    """
    test = df[df["session"].isin(test_sessions)]
    rest = df[~df["session"].isin(test_sessions)]
    dialogues = sorted(rest["dialogue"].unique())
    random.Random(seed).shuffle(dialogues)
    val_dialogues = set(dialogues[: max(1, round(len(dialogues) * val_fraction))])
    train = rest[~rest["dialogue"].isin(val_dialogues)]
    val = rest[rest["dialogue"].isin(val_dialogues)]
    if text_source == "asr":
        train, val = with_text(train, asr_texts), with_text(val, asr_texts)
    elif text_source == "both":
        train = pd.concat([train, with_text(train, asr_texts)], ignore_index=True)
        val = with_text(val, asr_texts)
    parts = {"train": train, "validation": val, "test": test}
    for name, part in parts.items():
        print(f"{name:<11} {len(part):>5} clips, {part['dialogue'].nunique():>3} dialogues, "
              f"sessions {sorted(part['session'].unique())}")
    return DatasetDict({name: Dataset.from_pandas(part[["file", "text"] + LABEL_COLS].reset_index(drop=True))
                        for name, part in parts.items()})


def tokenize(raw: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length)
        enc["labels"] = [[float(v), float(a), float(d)]
                         for v, a, d in zip(batch["valence"], batch["arousal"], batch["dominance"])]
        return enc

    return raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)


# ------------------------------------------------------------- loss/metrics

def ccc_per_dim(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Concordance correlation per output dimension, computed over the batch."""
    mp, ml = preds.mean(0), labels.mean(0)
    vp, vl = preds.var(0, unbiased=False), labels.var(0, unbiased=False)
    cov = ((preds - mp) * (labels - ml)).mean(0)
    return 2 * cov / (vp + vl + (mp - ml) ** 2 + 1e-8)


class VADTrainer(Trainer):
    def __init__(self, *args, loss_name: str = "ccc", **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        preds = outputs.logits.float()
        mse = torch.nn.functional.mse_loss(preds, labels)
        ccc_loss = (1 - ccc_per_dim(preds, labels)).mean()
        loss = {"ccc": ccc_loss, "mse": mse, "ccc+mse": ccc_loss + mse}[self.loss_name]
        return (loss, outputs) if return_outputs else loss


def ccc_np(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    return float(2 * ((x - mx) * (y - my)).mean() / (x.var() + y.var() + (mx - my) ** 2))


def compute_metrics(eval_pred) -> dict:
    """EmoBank metrics (Pearson, MAE, ...) plus CCC per dimension."""
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    out = regression_metrics(preds, labels)
    cccs = [ccc_np(np.asarray(preds)[:, i], np.asarray(labels)[:, i]) for i in range(len(LABEL_COLS))]
    out.update({f"ccc_{n}": c for n, c in zip(LABEL_COLS, cccs)})
    out["ccc_mean"] = float(np.mean(cccs))
    return out


# ------------------------------------------------------------------ main

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print("transformers:", transformers.__version__)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (training on CPU will be slow)")

    df = load_iemocap(args.iemocap_dir)
    asr_texts = None
    if args.text_source != "ref":
        if not args.asr_cache.exists():
            raise SystemExit(f"{args.asr_cache} not found - run Parakeet-to-VAD/evaluate_iemocap.py first")
        asr_texts = load_asr_texts(args.asr_cache)
        print(f"Training text: {args.text_source} (Parakeet transcripts from {args.asr_cache})")
    raw = split(df, args.test_sessions, args.val_fraction, args.seed, args.text_source, asr_texts)

    print(f"Starting from: {args.init_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.init_model, **_no_mistral_regex_patch())
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init_model, num_labels=len(LABEL_COLS), problem_type="regression")
    model = model.float()  # fp32 master weights for mixed precision
    model.config.id2label = {i: n for i, n in enumerate(LABEL_COLS)}
    model.config.label2id = {n: i for i, n in enumerate(LABEL_COLS)}
    data = tokenize(raw, tokenizer, args.max_length)

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
        metric_for_best_model="ccc_mean",
        greater_is_better=True,
        logging_steps=25,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
        seed=args.seed,
    )
    trainer = VADTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        loss_name=args.loss,
    )

    # Before training: the starting model on the test session (the baseline to beat)
    before = trainer.predict(data["test"], metric_key_prefix="test").metrics
    trainer.train()
    plot_training_curves(trainer.state.log_history, os.path.join(args.output_dir, "training_curves.png"))

    pred_out = trainer.predict(data["test"], metric_key_prefix="test")
    after = pred_out.metrics
    keys = [f"test_{m}_{n}" for m in ("pearson", "ccc", "mae") for n in LABEL_COLS]
    table = pd.DataFrame({"before": [before[k] for k in keys], "after": [after[k] for k in keys]},
                         index=[k.removeprefix("test_") for k in keys])
    print(f"\nTest sessions {args.test_sessions} (real transcripts; MAE on the 0-1 scale):")
    print(table.round(3).to_string())

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump({"init_model": str(args.init_model), "loss": args.loss, "text_source": args.text_source,
                   "test_sessions": args.test_sessions,
                   "before": before, "after": after}, f, indent=2)

    test_preds, test_labels = from_unit(pred_out.predictions), from_unit(pred_out.label_ids)
    plot_predictions(test_preds, test_labels, os.path.join(args.output_dir, "test_scatter.png"))
    plot_residuals(test_preds, test_labels, LABEL_COLS, os.path.join(args.output_dir, "test_residuals.png"))
    pd.DataFrame({"file": raw["test"]["file"], "text": raw["test"]["text"],
                  **{n: test_labels[:, i].round(3) for i, n in enumerate(LABEL_COLS)},
                  **{f"pred_{n}": test_preds[:, i].round(3) for i, n in enumerate(LABEL_COLS)}}
                 ).to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
