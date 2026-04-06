# Project 04 — Fraud Detection

Fraud detection on highly imbalanced transaction data.

## Why PR-AUC?
Accuracy is misleading when fraud is rare.
PR-AUC focuses on performance for the positive (fraud) class.

## Threshold Strategy
Default thresholds (0.5) underperform.
Tuning the threshold improved F1 and recall significantly.

## Key Insight
Cost-sensitive decisions matter more than raw accuracy in fraud systems.