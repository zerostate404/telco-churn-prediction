"""Sanity tests for src/interpret.py."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import src.config as cfg
from src.load_data import load_raw_data
from src.preprocess import clean_data, split_features_target
from src.train import build_logistic_pipeline, build_random_forest_pipeline
from src.interpret import (
    get_feature_names,
    logistic_coefficients,
    random_forest_importances,
)


@pytest.fixture(scope="module")
def fitted_models():
    clean = clean_data(load_raw_data())
    X, y = split_features_target(clean)
    X_tr, _, y_tr, _ = train_test_split(
        X,
        y,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )
    lr = build_logistic_pipeline().fit(X_tr, y_tr)
    rf = build_random_forest_pipeline().fit(X_tr, y_tr)
    return lr, rf


def test_feature_names_count_matches_coef(fitted_models):
    lr, _ = fitted_models
    names = get_feature_names(lr)
    coefs = lr.named_steps["classifier"].coef_[0]
    assert len(names) == len(coefs)


def test_logistic_coef_table_sorted(fitted_models):
    lr, _ = fitted_models
    df = logistic_coefficients(lr)
    abs_coefs = df["coefficient"].abs().to_numpy()
    assert (np.diff(abs_coefs) <= 1e-12).all(), "table must be sorted by |coef| desc"
    assert (df["odds_ratio"] > 0).all()


def test_odds_ratio_matches_exp(fitted_models):
    lr, _ = fitted_models
    df = logistic_coefficients(lr)
    assert np.allclose(df["odds_ratio"], np.exp(df["coefficient"]))


def test_rf_importances_sum_to_one(fitted_models):
    _, rf = fitted_models
    df = random_forest_importances(rf)
    assert df["importance"].sum() == pytest.approx(1.0, abs=1e-6)


def test_known_driver_contract_month_to_month(fitted_models):
    """Contract_Month-to-month is a top churn driver in every Telco model.
    If this fails, something is broken upstream (features, encoding, split)."""
    lr, _ = fitted_models
    df = logistic_coefficients(lr).head(10)
    assert "Contract_Month-to-month" in df["feature"].values
