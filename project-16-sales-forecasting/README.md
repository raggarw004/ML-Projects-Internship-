# Project 16 — Sales Forecasting

Time-series regression using lag features and rolling statistics.

## Feature Engineering
- Lag features (1, 7, 14 days)
- Rolling mean and standard deviation

## Models
- Baseline: Linear Regression
- Improved: Random Forest

## Forecast Results
Tree-based models capture non-linear seasonality better than linear baselines.

## Key Insight
Time-series problems require respecting temporal order — random shuffles break causality.