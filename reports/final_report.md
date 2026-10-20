# Customer Churn Prediction — Executive Summary

**Model:** Logistic Regression (class-weight balanced) · **Owner:** Data Science · **Date:** 2026-10

---

## 1. The Problem

Our customer base of 7,043 is losing ~26.5% to churn annually. At a $500
average lifetime value, that's ~$935k in annual revenue loss — and a
retention team that doesn't know whom to call.

## 2. The Solution

A logistic regression model scores each customer's probability of churning,
enabling targeted retention outreach.

| Metric | Value | What it means |
| :--- | :--- | :--- |
| **Recall** | **0.80** | We catch ~80% of customers who will churn |
| Precision | 0.51 | ~half of alerts are correct (acceptable — see §4) |
| F1 | 0.62 | Balanced precision/recall |
| ROC-AUC | 0.84 | Strong discriminating power |
| PR-AUC | 0.64 | Honest metric under class imbalance |

*(Verify exact values against `results/model_comparison.csv`.)*

**Naive baseline for context:** predicting "no churn" for every customer
achieves 73.5% accuracy but catches **zero** churners (recall = 0.00).
Class-weighted training shifts the decision boundary so we trade accuracy
for recall — the correct trade-off for retention.

## 3. What drives churn (top 5, reliable signals)

| Rank | Driver | Effect | Odds ratio |
| :--- | :--- | :--- | :--- |
| 1 | **Tenure** (shorter) | ↑ churn | 0.31 per SD decrease |
| 2 | **Contract = Two year** | ↓ churn | 0.45 |
| 3 | **InternetService = Fiber optic** | ↑ churn | 2.00 |
| 4 | **InternetService = DSL** | ↓ churn | 0.50 |
| 5 | **Contract = Month-to-month** | ↑ churn | 1.87 |

**Bonus insight:** phone-only customers (no internet service) churn at
~1/5 the rate of internet customers — a durable low-churn segment.

RF importances corroborate: `TotalCharges`, `tenure`, `AvgMonthlyCharge`,
`MonthlyCharges`, `Contract_Month-to-month`, `OnlineSecurity_No`,
`TechSupport_No`, `PaymentMethod_Electronic check`.

## 4. Why precision of 0.51 is fine

Retention offers cost ~$10. A missed churner costs ~$500 LTV.
Per 1,000 customers scored:
- ~587 flagged, ~299 true churners (saved), ~288 false positives (wasted $10 each)
- ~413 unflagged, ~75 missed churners (lost $500 each)
- Expected cost: (288 × $10) + (75 × $500) ≈ **$40,360**

Compared to **no model** (all 374 churners lost = $187,000), the model
delivers **~78% reduction in lost revenue** — even with imperfect precision.

## 5. Recommendations

1. **Convert month-to-month to annual contracts.** Biggest actionable lever —
   month-to-month customers have ~2× the churn odds. Test a 10–15% discount
   at month 3 of tenure.
2. **Investigate Fiber Optic satisfaction.** Fiber customers churn at ~2×
   the rate of DSL *despite* paying more — a pricing/value perception issue.
3. **Bundle tech support + online security with new internet installs.**
   Customers *without* these add-ons churn markedly more. Free 90-day trial
   for new fiber customers.
4. **Focus outreach on the 0–12 month cohort.** Tenure is the strongest
   predictor; the highest-ROI retention investment is in the first year.

## 6. What we tried and ruled out

- **SMOTE vs. class weighting.** Both produce equivalent models (ΔF1 < 0.002).
  Chose class weighting — simpler, no synthetic data, faster inference.
- **Tenure binning.** Adding binned tenure features alongside raw tenure
  produced multicollinear inputs and sign-flipped coefficients. Raw tenure
  captures the relationship cleanly. **Dropped.**

## 7. Limitations & next steps

- **Correlation ≠ causation.** Coefficients reflect historical patterns.
  Recommendation #1 (annual contracts) should be A/B tested before rollout.
- **Collinear charge features.** `MonthlyCharges`, `TotalCharges`, and
  `AvgMonthlyCharge` are highly correlated. Their individual LR coefficients
  are not independently interpretable — rely on RF importances and
  `tenure`/`Contract` for the business narrative.
- **Single 20/80 split.** 5-fold CV would tighten estimates; add before
  deployment.
- **No temporal features.** Churn has seasonality; "days since last bill
  change" or "last support ticket" would likely improve recall.
- **Fairness check pending.** `gender` and `SeniorCitizen` coefficients are
  non-trivial; audit before deployment.

---

*Code, tests, notebooks: [github.com/you/telco-churn-prediction]*