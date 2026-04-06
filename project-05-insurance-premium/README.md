# Project 05 — Insurance Premium Prediction

This project predicts insurance premiums using demographic and lifestyle data.

## Feature Engineering
- Age bands (18–30, 31–45, 46–60, 60+)
- BMI bands (normal → severely obese)

## Models
- Baseline: Linear Regression
- Improved: Gradient Boosting Regressor

## Segment-wise Error Analysis
MAE was highest for:
- Older customers (60+) with high BMI
- Obese smokers across all age groups

This suggests the model struggles most where risk factors compound.

## Business Insight
Segment-level error analysis is critical for pricing fairness and regulatory review.