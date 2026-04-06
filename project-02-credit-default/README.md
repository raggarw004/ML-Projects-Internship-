# Project 02 — Credit Default Prediction

Binary classification project to predict loan default with imbalanced data.

## Models
- Baseline: Logistic Regression (class-weighted)
- Improved: Gradient Boosting

## Evaluation
- F1-score
- ROC-AUC
- Threshold tuning for business decisions

## Results
See `reports/metrics.json` for final metrics.

## Key Insight
Accuracy is misleading for imbalanced data — F1 and ROC-AUC better capture model quality.