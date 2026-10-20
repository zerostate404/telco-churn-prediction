"""Evaluation utilities for binary classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import src.config as cfg


def evaluate_classifier(model, X, y, threshold: float = 0.5) -> dict:
    """Compute the standard classification metric suite at a probability threshold.

    Notes:
    - `roc_auc` and `pr_auc` are threshold-independent; the rest use `threshold`.
    - `pr_auc` (average precision) is the right "area" metric for imbalanced data;
      ROC-AUC tends to be optimistic when the minority class is small.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 2),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
    }


def print_classification_report(model, X, y, target_names=("No churn", "Churn")):
    """Print sklearn's classification_report plus a confusion matrix."""
    pred = model.predict(X)
    print(classification_report(y, pred, target_names=list(target_names), digits=4))
    cm = confusion_matrix(y, pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"    {'No':>6} {'Yes':>6}")
    print(f"No  {cm[0,0]:>6} {cm[0,1]:>6}")
    print(f"Yes {cm[1,0]:>6} {cm[1,1]:>6}")


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------
def sweep_thresholds(model, X, y, thresholds=None) -> pd.DataFrame:
    """Return a DataFrame of metrics at each threshold in [0.05, 0.95]."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    proba = model.predict_proba(X)[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def find_best_f1_threshold(model, X, y) -> tuple[float, dict]:
    """Return the threshold that maximizes F1 on the given data."""
    df = sweep_thresholds(model, X, y)
    best = df.loc[df["f1"].idxmax()]
    return float(best["threshold"]), best.to_dict()


# Plots


def plot_roc_pr_curves(models: dict, X, y, save_dir=cfg.FIGURES_DIR):
    """Side-by-side ROC + PR curves for multiple models."""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

        prec, rec, _ = precision_recall_curve(y, proba)
        pr_auc = average_precision_score(y, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    axes[0].set(
        xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves"
    )
    axes[0].legend(loc="lower right")

    axes[1].axhline(
        y.mean(),
        color="k",
        linestyle="--",
        alpha=0.3,
        label=f"Baseline ({y.mean():.2f})",
    )
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    out = save_dir / "roc_pr_curves.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


def plot_threshold_analysis(model, X, y, name: str, save_dir=cfg.FIGURES_DIR):
    """Precision/recall/F1 vs threshold, with F1-optimal threshold marked."""
    save_dir.mkdir(parents=True, exist_ok=True)
    df = sweep_thresholds(model, X, y)
    best_t, best_row = find_best_f1_threshold(model, X, y)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision", marker="o", ms=3)
    ax.plot(df["threshold"], df["recall"], label="Recall", marker="o", ms=3)
    ax.plot(df["threshold"], df["f1"], label="F1", marker="o", ms=3, lw=2)

    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.6, label="Default (0.5)")
    ax.axvline(
        best_t,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best F1 = {best_row['f1']:.3f} @ t={best_t}",
    )

    ax.set(
        xlabel="Threshold",
        ylabel="Score",
        title=f"Threshold Analysis — {name}",
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = save_dir / f"threshold_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")
    print(
        f"Best F1 = {best_row['f1']:.4f} at threshold={best_t} "
        f"(precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f})"
    )
