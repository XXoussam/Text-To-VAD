"""Predict Valence / Arousal / Dominance for text with a fine-tuned model.

Examples:
    python predict.py "What an incredible pentakill!" "Blue team secures dragon."
    python predict.py --file sentences.txt --scale emobank --output preds.csv
    python predict.py            # runs a few demo sentences
"""

import argparse

import pandas as pd

from text_to_vad.config import MAX_LENGTH, OUTPUT_DIR
from text_to_vad.inference import VADPredictor

DEMO_TEXTS = [
    "What an unbelievable teamfight!",
    "What an incredible pentakill!",
    "Blue team secures dragon.",
    "The fight is slowly developing.",
    "They threw the game, this is a disaster.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("texts", nargs="*", help="Sentences to score")
    p.add_argument("--file", help="Text file with one sentence per line")
    p.add_argument("--model-dir", default=OUTPUT_DIR, help="Local dir or Hugging Face Hub repo id")
    p.add_argument("--scale", choices=["unit", "emobank"], default="unit",
                   help="'unit' → 0–1, 'emobank' → original 1–5")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--output", help="Optional CSV path to write predictions")
    return p.parse_args()


def main():
    args = parse_args()

    texts = list(args.texts)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        texts = DEMO_TEXTS

    predictor = VADPredictor(args.model_dir, max_length=args.max_length)
    result = predictor.predict(texts, scale=args.scale, batch_size=args.batch_size)

    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(result)
    if args.output:
        result.rename_axis("text").to_csv(args.output)
        print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
