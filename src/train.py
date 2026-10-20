"""Model pipelines and training utilities for telco churn.

Design: one preprocessor, plugged into two different classifier Pipelines.
Any change to feature handling flows through both models automatically.
"""

from __future__ import annotations

import time

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.config as cfg

# Preprocessor


def build_preprocessor() -> ColumnTransformer:
    """Build the feature ColumnTransformer.

    - Numeric (raw + derived) → StandardScaler
    - Binary (already int8)   → passthrough (no scaling needed for 0/1)
    - Multi-categorical       → OneHotEncoder (string columns)
    """
    numeric_cols = cfg.NUMERIC_FEATURES + cfg.DERIVED_NUMERIC_FEATURES
    binary_cols = cfg.BINARY_FEATURES + cfg.PRE_ENCODED_BINARY_FEATURES
    cat_cols = cfg.MULTICAT_FEATURES + cfg.DERIVED_CATEGORICAL_FEATURES

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", binary_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",  # safety net: unlisted columns vanish
        verbose_feature_names_out=False,
    )


# Pipelines


def build_logistic_pipeline() -> Pipeline:
    """Logistic Regression baseline with balanced class weights."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=cfg.RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Random Forest with balanced class weights."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=cfg.RANDOM_STATE,
                ),
            ),
        ]
    )


# Training


def train_model(pipeline: Pipeline, X_train, y_train) -> tuple[Pipeline, float]:
    """Fit the pipeline in place. Returns (fitted_pipeline, train_seconds)."""
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    return pipeline, time.perf_counter() - t0
