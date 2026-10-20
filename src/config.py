"""Project-wide configuration: paths and constants."""

from pathlib import Path

# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"

RAW_DATA_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ---------- Reproducibility ----------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------- Target ----------
TARGET_COL = "Churn"

# ---------- Column groups ----------
ID_COL = "customerID"

# Raw numeric features
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# String yes/no (and gender) → 0/1
BINARY_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Already 0/1 in the raw file — cast only
PRE_ENCODED_BINARY_FEATURES = ["SeniorCitizen"]

# Multi-category columns → one-hot encode
MULTICAT_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# ---------- Derived features (created in preprocess.py) ----------
DERIVED_NUMERIC_FEATURES = ["TotalServices", "AvgMonthlyCharge"]
DERIVED_CATEGORICAL_FEATURES = []  # tenure bins dropped: collinear with raw tenure

# ---------- Feature engineering parameters ----------
TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
