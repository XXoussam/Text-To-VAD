"""Evaluate a saved VAD model on an EmoBank split.

Example:
    python evaluate_model.py --model-dir ./deberta_vad --split test
"""

import argparse
import json
import os

import pandas as pd

from text_to_vad.config import EMOBANK_URL, LABEL_COLS, MAX_LENGTH, OUTPUT_DIR, from_unit
from text_to_vad.data import SPLITS, load_emobank
from text_to_vad.inference import VADPredictor
from text_to_vad.metrics import plot_predictions, regression_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=OUTPUT_DIR)
    p.add_argument("--data", default=EMOBANK_URL, help="EmoBank CSV (URL or local path)")
    p.add_argument("--split", default="test", choices=list(SPLITS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--plot", default=None, help="Where to save the scatter plot (default: <model-dir>/<split>_scatter.png)")
    return p.parse_args()


def main():
    args = parse_args()

    df = load_emobank(args.data)
    part = df[df["split"] == SPLITS[args.split]]
    print(f"Evaluating on {len(part)} '{args.split}' sentences")

    predictor = VADPredictor(args.model_dir, max_length=args.max_length)
    preds = predictor.predict_array(part["text"].tolist(), batch_size=args.batch_size)
    labels = part[LABEL_COLS].to_numpy()

    metrics = {f"{args.split}_{k}": v for k, v in regression_metrics(preds, labels).items()}
    print(pd.Series(metrics).round(4).to_string())
    with open(os.path.join(args.model_dir, f"{args.split}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_path = args.plot or os.path.join(args.model_dir, f"{args.split}_scatter.png")
    plot_predictions(from_unit(preds), from_unit(labels), plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
