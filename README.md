# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn for a telecom company. It walks through the complete data-science workflow — exploratory data analysis, statistical hypothesis testing, data cleaning, feature engineering, class-imbalance handling, and model training — using Logistic Regression and Random Forest classifiers. Special emphasis is placed on model interpretability and translating results into business recommendations for a retention team.

**Stack:** Python · pandas · scikit-learn · imbalanced-learn · scipy · seaborn · matplotlib

**Skills demonstrated:** EDA · hypothesis testing · feature engineering · missing-value and outlier handling · class imbalance (SMOTE, class weights) · model evaluation (precision/recall/F1/ROC-AUC) · model interpretation

---

## Problem

A telecom company loses approximately **26.5% of customers annually** to churn. On the 7,043-customer dataset, that's approximately 1,871 churned customers per year. At $500 average lifetime value, that's approximately $935k in lost revenue — and a retention team that doesn't know whom to call.

This project builds a model that scores each customer's probability of churning, enabling **targeted retention outreach** with measurable ROI.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always-"No" baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| **Logistic Regression (class-weighted)** | 0.740 | 0.507 | **0.800** | 0.620 | 0.843 | 0.643 |
| Random Forest (class-weighted) | 0.759 | 0.540 | 0.626 | 0.580 | 0.823 | 0.615 |

**Headline:** The logistic regression model catches **80% of customers who actually churn** — worth an estimated **78% reduction in lost revenue** versus having no model — while the always-"No" baseline catches zero churners despite higher raw accuracy. This demonstrates why accuracy is the wrong metric under class imbalance: precision, recall, and PR-AUC tell the real story.

**Additional experiments:**

- **Threshold tuning:** The F1-optimal threshold is 0.60 for LR and 0.35 for RF. LR's recall and precision are nearly identical at 0.50 (its class-weight loss already shifts the boundary); RF benefits from the lower threshold, recovering its recall gap. We ship at 0.50 for simplicity, but the notebook demonstrates the tuning workflow.
- **SMOTE vs. class weights:** Both approaches produce equivalent models (ΔF1 < 0.002, ΔROC-AUC < 0.003). We chose class weighting — simpler, no synthetic data, faster inference, and no `imblearn` dependency at serving time.
- **Tenure binning:** Initial feature engineering added binned tenure features (0–12, 13–24, …) alongside raw tenure. Coefficient inspection revealed multicollinearity producing sign-flipped coefficients. **Dropped** — raw tenure captures the relationship cleanly.

## Top churn drivers

| Rank | Driver | Effect | Odds ratio (LR) |
| ---: | :--- | :--- | ---: |
| 1 | **Tenure (shorter)** | Increases churn | 0.31 per SD decrease |
| 2 | **Contract = Two year** | Decreases churn | 0.45 |
| 3 | **InternetService = Fiber optic** | Increases churn | 2.00 |
| 4 | **InternetService = DSL** | Decreases churn | 0.50 |
| 5 | **Contract = Month-to-month** | Increases churn | 1.87 |

Additional signal from Random Forest importances corroborates: no Online Security, no Tech Support, and electronic-check payment all predict churn. Phone-only customers (no internet service) churn at approximately one-fifth the rate of internet customers — a durable low-churn segment.

## Visualizations

![ROC and PR curves](figures/roc_pr_curves.png)

*Left: ROC curves — LR dominates slightly (AUC 0.843 vs 0.823). Right: PR curves — the honest view under imbalance. LR sits above RF in the high-recall region where a retention model operates.*

![LR coefficients](figures/lr_coefficients.png)

*Top 20 LR drivers by absolute coefficient. Red = increases churn, blue = decreases. Tenure and contract type dominate.*

![RF feature importances](figures/rf_importances.png)

*Top 20 RF importances by Gini. TotalCharges, tenure, and monthly-charge variants lead — the "customer value" cluster.*

## Business value

**Why precision of 0.51 is acceptable:** Retention offers cost approximately $10; a missed churner costs approximately $500 in lifetime value. Per 1,000 customers scored at the current operating point:

- Approximately 587 flagged → 299 true churners (saved) + 288 false positives (approximately $2,880 wasted).
- Approximately 413 unflagged → 75 real churners missed (approximately $37,500 lost).
- **Total expected cost: approximately $40,360.**

Compared to **no model** (approximately $187,000 lost on approximately 374 churners), the model delivers an estimated **78% reduction in lost revenue**, even with imperfect precision.

## Recommendations for the retention team

1. **Convert month-to-month to annual contracts.** Biggest actionable lever — month-to-month customers have approximately twice the churn odds. A/B test a 10–15% discount offered at month 3 of tenure.
2. **Investigate Fiber Optic satisfaction.** Fiber customers churn at approximately twice the rate of DSL despite paying more. Retention risk and revenue risk make this a priority for customer-satisfaction investigation.
3. **Bundle tech support and online security with new internet installs.** Customers without these add-ons churn markedly more. Consider a free 90-day trial for new fiber customers.
4. **Focus outreach on the 0–12 month cohort.** Tenure is the single strongest predictor; the highest-ROI retention investment is in the first year.

Full narrative: [`reports/final_report.md`](reports/final_report.md)

## Project structure

```text
telco-churn-prediction/
├── data/
│   ├── raw/                     # Original Telco CSV (user-supplied)
│   └── processed/               # Derived artifacts
├── notebooks/
│   ├── 01_eda.ipynb             # EDA, hypothesis tests, schema validation
│   ├── 02_preprocessing.ipynb   # Validation of cleaning pipeline
│   ├── 03_modeling.ipynb        # Train LR + RF, curves, threshold analysis
│   └── 04_interpretation.ipynb  # Coefficients, importances, narrative
├── src/
│   ├── config.py                # Paths, column groups, feature definitions
│   ├── load_data.py             # I/O only
│   ├── preprocess.py            # Stateless cleaning + feature engineering
│   ├── train.py                 # sklearn pipelines
│   ├── evaluate.py              # Metric suite, curves, threshold sweep
│   └── interpret.py             # Coefficients, importances, comparison
├── tests/                       # pytest: 22 tests
│   ├── test_preprocess.py       # 9 tests
│   ├── test_train.py            # 8 tests
│   └── test_interpret.py        # 5 tests
├── figures/                     # Saved plots (regenerated by notebooks)
├── results/                     # Fitted .joblib models, metric CSVs
├── reports/
│   └── final_report.md          # Executive summary
├── requirements.txt
├── .gitignore
└── README.md
```

**Design principles:**

- **Stateless cleaning** lives in `preprocess.py`; **stateful transforms** (scaling, one-hot) live inside the sklearn Pipeline in `train.py`. This prevents train/test leakage.
- **Single preprocessor** feeds both LR and RF — comparison is apples-to-apples.
- **Fail-loud assertions** in the notebooks guard against stale pickles and silent NaN generation (a class of bug we hit during development; see "Lessons learned" below).
- **Tests assert invariants**, not implementation details — e.g., "churn rate preserved across cleaning," "no `tenure_group` features in the fitted model."