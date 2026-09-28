"""Load a fine-tuned model and predict VAD scores for raw text."""

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import LABEL_COLS, MAX_LENGTH, OUTPUT_DIR, from_unit


def _no_mistral_regex_patch() -> dict:
    """transformers 4.57.3–4.x flags *any* locally saved tokenizer as a broken Mistral one
    and warns to pass `fix_mistral_regex=True`, which would swap DeBERTa's pre-tokenizer
    for Mistral's. Passing False explicitly silences that false positive and keeps ours."""
    from transformers import PreTrainedTokenizerBase

    return {"fix_mistral_regex": False} if hasattr(PreTrainedTokenizerBase, "_patch_mistral_regex") else {}


class VADPredictor:
    def __init__(self, model_dir: str = OUTPUT_DIR, device: Optional[str] = None,
                 max_length: int = MAX_LENGTH):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **_no_mistral_regex_patch())
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_array(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Raw 0–1 predictions (clipped), shape (n, 3)."""
        texts = list(texts)
        chunks = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[start:start + batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            chunks.append(self.model(**enc).logits.float().cpu().numpy())
        out = np.concatenate(chunks) if chunks else np.empty((0, len(LABEL_COLS)))
        return np.clip(out, 0.0, 1.0)

    def predict(self, texts: Union[str, Iterable[str]], scale: str = "unit",
                batch_size: int = 32) -> pd.DataFrame:
        """scale='unit' → 0–1, scale='emobank' → original 1–5."""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out = self.predict_array(texts, batch_size=batch_size)
        if scale == "emobank":
            out = from_unit(out)
        return pd.DataFrame(out, columns=LABEL_COLS, index=texts).round(3)
