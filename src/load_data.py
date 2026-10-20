"""Data ingestion utilities."""

import pandas as pd

import src.config as cfg


def load_raw_data(path=cfg.RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw Telco churn CSV from data/raw/."""
    return pd.read_csv(path)
