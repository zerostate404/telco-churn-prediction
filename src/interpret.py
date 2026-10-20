"""Model interpretation utilities: coefficients, feature importances, plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.config as cfg


# ---------------------------------------------------------------------------
# Feature name extraction
# ---------------------------------------------------------------------------
def get_feature_names(pipeline) -> np.ndarray:
    """Get feature names after the ColumnTransformer has expanded categories.

    Because build_preprocessor() uses verbose_feature_names_out=False, names
    come out clean (e.g. 'tenure', 'InternetService_Fiber optic').
    """
    return pipeline.named_steps["preprocessor"].get_feature_names_out()


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
def logistic_coefficients(pipeline) -> pd.DataFrame:
    """Return a feature table with coefficient, odds ratio, and |coef|.

    Interpretation of odds_ratio for a feature f:
      - OR > 1 → one-unit increase in f multiplies churn odds by OR
      - OR < 1 → decreases churn odds
      - OR = 1 → no effect (rare in practice)
    For scaled numerics, "one unit" = one standard deviation.
    """
    names = get_feature_names(pipeline)
    coefs = pipeline.named_steps["classifier"].coef_[0]
    df = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    df["abs_coef"] = df["coefficient"].abs()
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return df.drop(columns="abs_coef")


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def random_forest_importances(pipeline) -> pd.DataFrame:
    """Return a feature table with Gini importances, sorted descending."""
    names = get_feature_names(pipeline)
    importances = pipeline.named_steps["classifier"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_coefficients(coef_df: pd.DataFrame, top_n: int = 20, save_dir=cfg.FIGURES_DIR):
    """Horizontal bar chart of top N LR coefficients (red = increases churn)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    top = coef_df.head(top_n).iloc[::-1]
    colors = ["crimson" if c > 0 else "steelblue" for c in top["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.32))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Coefficient (log-odds) — red ↑churn, blue ↓churn",
        title=f"Top {top_n} drivers — Logistic Regression",
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = save_dir / "lr_coefficients.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


def plot_importances(imp_df: pd.DataFrame, top_n: int = 20, save_dir=cfg.FIGURES_DIR):
    """Horizontal bar chart of top N RF importances."""
    save_dir.mkdir(parents=True, exist_ok=True)
    top = imp_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.32))
    ax.barh(top["feature"], top["importance"], color="seagreen")
    ax.set(
        xlabel="Gini importance",
        title=f"Top {top_n} drivers — Random Forest",
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = save_dir / "rf_importances.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


def compare_top_features(
    coef_df: pd.DataFrame, imp_df: pd.DataFrame, top_n: int = 15
) -> pd.DataFrame:
    """Side-by-side table: top-N features from each model, with ranks."""
    lr_top = coef_df.head(top_n).copy()
    lr_top["lr_rank"] = range(1, len(lr_top) + 1)
    lr_top = lr_top[["feature", "coefficient", "lr_rank"]]

    rf_top = imp_df.head(top_n).copy()
    rf_top["rf_rank"] = range(1, len(rf_top) + 1)
    rf_top = rf_top[["feature", "importance", "rf_rank"]]

    merged = lr_top.merge(rf_top, on="feature", how="outer")
    return merged.sort_values("rf_rank", na_position="last").reset_index(drop=True)
