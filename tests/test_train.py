"""Unit tests for src/train.py and src/evaluate.py."""

import pytest
from sklearn.model_selection import train_test_split

import src.config as cfg
from src.load_data import load_raw_data
from src.preprocess import clean_data, split_features_target
from src.train import (
    build_logistic_pipeline,
    build_random_forest_pipeline,
    train_model,
)
from src.evaluate import evaluate_classifier


@pytest.fixture(scope="module")
def data():
    clean = clean_data(load_raw_data())
    X, y = split_features_target(clean)
    return train_test_split(
        X,
        y,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )


@pytest.fixture(scope="module")
def trained_lr(data):
    X_train, _, y_train, _ = data
    return train_model(build_logistic_pipeline(), X_train, y_train)[0]


def test_lr_pipeline_fits(trained_lr):
    assert hasattr(trained_lr, "predict")


def test_lr_predict_shape(trained_lr, data):
    _, X_test, _, _ = data
    assert trained_lr.predict(X_test).shape == (len(X_test),)


def test_lr_predict_proba_shape(trained_lr, data):
    _, X_test, _, _ = data
    assert trained_lr.predict_proba(X_test).shape == (len(X_test), 2)


def test_metrics_in_range(trained_lr, data):
    _, X_test, _, y_test = data
    m = evaluate_classifier(trained_lr, X_test, y_test)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert 0.0 <= m[key] <= 1.0, f"{key} out of range: {m[key]}"


def test_lr_beats_majority_baseline(trained_lr, data):
    """ROC-AUC must beat the 0.5 coin-flip baseline decisively."""
    _, X_test, _, y_test = data
    m = evaluate_classifier(trained_lr, X_test, y_test)
    assert m["roc_auc"] > 0.75, f"ROC-AUC too low: {m['roc_auc']}"


def test_rf_pipeline_fits(data):
    X_train, X_test, y_train, _ = data
    rf, _ = train_model(build_random_forest_pipeline(), X_train, y_train)
    assert rf.predict(X_test).shape == (len(X_test),)


from src.evaluate import sweep_thresholds, find_best_f1_threshold


def test_threshold_sweep_shape(trained_lr, data):
    _, X_test, _, y_test = data
    df = sweep_thresholds(trained_lr, X_test, y_test)
    assert set(df.columns) == {"threshold", "precision", "recall", "f1"}
    assert len(df) == 19  # np.arange(0.05, 0.96, 0.05) has 19 entries


def test_best_f1_threshold_returns_valid_tuple(trained_lr, data):
    _, X_test, _, y_test = data
    t, row = find_best_f1_threshold(trained_lr, X_test, y_test)
    assert 0.0 < t < 1.0
    assert row["f1"] >= 0.4  # must beat a trivial model on Telco
