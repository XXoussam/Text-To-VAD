"""Regression metrics (overall and per VAD dimension) and prediction plots."""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import LABEL_COLS


def regression_metrics(preds, labels) -> dict:
    """MAE, RMSE, R² overall + Pearson r and MAE per dimension."""
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    out = {
        "mae": float(mean_absolute_error(labels, preds)),
        "rmse": float(np.sqrt(mean_squared_error(labels, preds))),
        "r2": float(r2_score(labels, preds)),
    }
    rs = []
    for i, name in enumerate(LABEL_COLS):
        r = pearsonr(labels[:, i], preds[:, i])[0]
        rs.append(r)
        out[f"pearson_{name}"] = float(r)
        out[f"mae_{name}"] = float(mean_absolute_error(labels[:, i], preds[:, i]))
    out["pearson_mean"] = float(np.mean(rs))
    return out


def compute_metrics(eval_pred) -> dict:
    """`Trainer`-compatible wrapper around `regression_metrics`."""
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    return regression_metrics(preds, labels)


def plot_predictions(preds, labels, path: str, lo: float = 1.0, hi: float = 5.0) -> None:
    """Save a predicted-vs-true scatter plot per dimension to `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, name in enumerate(LABEL_COLS):
        ax = axes[i]
        ax.scatter(labels[:, i], preds[:, i], s=6, alpha=0.4)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        r = pearsonr(labels[:, i], preds[:, i])[0]
        ax.set_title(f"{name}  (r = {r:.3f})")
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
