"""Unit tests for src/preprocess.py."""

import pandas as pd
import pytest

from src.config import TARGET_COL, ID_COL
from src.load_data import load_raw_data
from src.preprocess import clean_data, split_features_target


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    return clean_data(load_raw_data())


def test_no_missing_values(cleaned):
    assert cleaned.isna().sum().sum() == 0


def test_id_dropped(cleaned):
    assert ID_COL not in cleaned.columns


def test_target_is_binary_int(cleaned):
    assert cleaned[TARGET_COL].dtype.name == "int8"
    assert set(cleaned[TARGET_COL].unique()) == {0, 1}


def test_totalcharges_is_numeric(cleaned):
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_derived_features_present(cleaned):
    for col in ["tenure_group", "TotalServices", "AvgMonthlyCharge"]:
        assert col in cleaned.columns, f"missing derived feature {col}"


def test_total_services_range(cleaned):
    assert cleaned["TotalServices"].between(0, 8).all()


def test_avg_monthly_charge_finite(cleaned):
    assert cleaned["AvgMonthlyCharge"].notna().all()
    assert (cleaned["AvgMonthlyCharge"] >= 0).all()


def test_row_count_preserved(cleaned):
    """No rows dropped during cleaning."""
    assert len(cleaned) == 7043


def test_split_shapes(cleaned):
    X, y = split_features_target(cleaned)
    assert len(X) == len(y) == len(cleaned)
    assert TARGET_COL not in X.columns


def test_tenure_zero_captured_in_first_bin(cleaned):
    raw = load_raw_data()
    mask = raw["tenure"] == 0
    assert mask.sum() > 0
    assert cleaned.loc[mask.values, "tenure_group"].eq("0-12").all()


def test_tenure_group_has_no_nulls(cleaned):
    assert cleaned["tenure_group"].notna().all()
