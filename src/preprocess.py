"""Cleaning and feature engineering for the Telco churn dataset.

Design principle: everything in this module is STATELESS — no statistics
are fit from data. Stateful transforms (scaling, one-hot encoding) live in
the sklearn Pipeline in src/train.py to prevent data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.config as cfg


def _cast_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges ships as strings with blank entries for new customers.
    Cast to numeric; blanks become NaN (filled in the next step)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def _fill_missing_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Missing TotalCharges = tenure == 0 (new customers, never billed).
    Filling with 0.0 is semantically correct, not arbitrary."""
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features. Must run BEFORE binary encoding so service
    columns are still 'Yes'/'No'/'No internet service' strings."""
    df = df.copy()

    # 1) Count of subscribed services (0..8); only 'Yes' counts.
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["TotalServices"] = df[service_cols].eq("Yes").sum(axis=1).astype("int8")

    # 2) Average monthly charge; tenure == 0 → use MonthlyCharges.
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )
    return df


def _encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No (and Male/Female) → 1/0. Fails loudly on unseen values.
    SeniorCitizen is already int — just cast."""
    df = df.copy()
    yes_no = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}

    for col in cfg.BINARY_FEATURES:
        if col == cfg.TARGET_COL:
            continue
        mapping = gender_map if col == "gender" else yes_no
        unknown = set(df[col].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected values in {col!r}: {unknown}")
        df[col] = df[col].map(mapping).astype("int8")

    for col in cfg.PRE_ENCODED_BINARY_FEATURES:
        df[col] = df[col].astype("int8")

    return df


# Public API


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full stateless cleaning + feature engineering.

    Returns a DataFrame with:
      - no missing values
      - binary columns encoded as int8
      - multi-category columns left as strings (for OneHotEncoder)
      - derived features added
      - customerID dropped
      - Churn encoded as int8 (0/1)
    """
    df = df.copy()
    df = df.drop(columns=[cfg.ID_COL])
    df = _cast_total_charges(df)
    df = _fill_missing_total_charges(df)
    df = _engineer_features(df)
    df = _encode_binary(df)

    churn_vals = set(df[cfg.TARGET_COL].unique())
    if churn_vals != {"Yes", "No"}:
        raise ValueError(
            f"Unexpected {cfg.TARGET_COL} values before encoding: {churn_vals}. "
            f"Did you accidentally leave {cfg.TARGET_COL!r} in BINARY_FEATURES?"
        )
    df[cfg.TARGET_COL] = df[cfg.TARGET_COL].map({"Yes": 1, "No": 0}).astype("int8")
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned DataFrame into (X, y)."""
    y = df[cfg.TARGET_COL].astype("int8")
    X = df.drop(columns=[cfg.TARGET_COL])
    return X, y
