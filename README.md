# Text-to-VAD (DeBERTa-v3 on EmoBank)

Fine-tunes `microsoft/deberta-v3-base` to predict **Valence, Arousal, Dominance** from text, trained on [EmoBank](https://github.com/JULIELab/EmoBank) (~10k sentences, official train/dev/test split).

```
Input text → DeBERTa-v3-base → pooled [CLS] (768) → Linear(768 → 3) → [V, A, D]
```

Labels are rescaled from EmoBank's 1–5 to 0–1 for training. EmoBank scores cluster near neutral, so MAE looks small; **Pearson r** and **R²** are the more informative metrics.

## Layout

```
text_to_vad/        shared code
  config.py         constants + 1–5 ↔ 0–1 scaling
  data.py           load EmoBank, build/tokenize datasets
  metrics.py        MAE / RMSE / R² / Pearson + scatter plot
  inference.py      VADPredictor
train.py            fine-tune, evaluate on test, save model
evaluate_model.py   evaluate a saved model on any split
predict.py          score sentences from the CLI / a file
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
python -m pip install -U pip    # recent pip trusts the Windows certificate store (needed behind corporate proxies)
pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124   # GPU build, install first
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: 2.6.0+cu124 True
```

If pip fails with `CERTIFICATE_VERIFY_FAILED` for `download.pytorch.org`, your network intercepts HTTPS.
Upgrading pip (above) usually fixes it; otherwise add `--trusted-host download.pytorch.org` to the torch install.

## Usage

```bash
# Train (≈15–25 min on a T4). Writes model, test_metrics.json, test_scatter.png to ./deberta_vad
python train.py

# Evaluate a saved model
python evaluate_model.py --model-dir ./deberta_vad --split test

# Predict
python predict.py "What an incredible pentakill!" "Blue team secures dragon."
python predict.py --file sentences.txt --scale emobank --output preds.csv
```

Run `python <script>.py --help` for all options.
