"""Loading EmoBank and turning it into tokenized Hugging Face datasets."""

import pandas as pd
from datasets import Dataset, DatasetDict

from .config import EMOBANK_URL, LABEL_COLS, MAX_LENGTH, to_unit

# EmoBank split name → Hugging Face split name
SPLITS = {"train": "train", "validation": "dev", "test": "test"}


def load_emobank(source: str = EMOBANK_URL) -> pd.DataFrame:
    """Read EmoBank (URL or local CSV) with columns `id, split, V, A, D, text`.

    Adds `valence / arousal / dominance` columns rescaled from 1–5 to 0–1.
    """
    df = pd.read_csv(source)
    df = df.dropna(subset=["text"])  # one row has empty text
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    df["valence"] = to_unit(df["V"])
    df["arousal"] = to_unit(df["A"])
    df["dominance"] = to_unit(df["D"])
    return df


def build_datasets(df: pd.DataFrame) -> DatasetDict:
    """Keep EmoBank's own train / dev / test split."""
    def make_split(name):
        part = df[df["split"] == name][["text"] + LABEL_COLS].reset_index(drop=True)
        return Dataset.from_pandas(part)

    return DatasetDict({hf_name: make_split(eb_name) for hf_name, eb_name in SPLITS.items()})


def tokenize_datasets(raw_ds: DatasetDict, tokenizer, max_length: int = MAX_LENGTH) -> DatasetDict:
    """Tokenize text and pack `labels = [valence, arousal, dominance]`.

    Padding is left to `DataCollatorWithPadding` (per batch).
    """
    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length)
        enc["labels"] = [
            [float(v), float(a), float(d)]
            for v, a, d in zip(batch["valence"], batch["arousal"], batch["dominance"])
        ]
        return enc

    return raw_ds.map(
        preprocess,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )
