# Project 03 — Customer Churn Prediction

This project predicts customer churn using tabular business data.

## Models
- Baseline: Logistic Regression
- Improved: Random Forest

## Key Churn Drivers
From permutation importance, the strongest churn signals were:
1. Month-to-month contracts
2. High monthly charges
3. Low customer tenure

## Business Insight
Customers on short contracts with high charges are most likely to churn.
Retention strategies should prioritize contract upgrades and early engagement.

## Artifacts
- Churn driver chart: `reports/figures/churn_drivers.png`
- Metrics: `reports/metrics.json`