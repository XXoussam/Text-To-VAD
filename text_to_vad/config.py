"""Shared constants and label scaling helpers."""

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "./deberta_vad"
MAX_LENGTH = 128
SEED = 42
LABEL_COLS = ["valence", "arousal", "dominance"]

EMOBANK_URL = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"


def to_unit(x):
    """EmoBank 1–5 scale → 0–1."""
    return (x - 1.0) / 4.0


def from_unit(x):
    """0–1 scale → EmoBank 1–5."""
    return x * 4.0 + 1.0
