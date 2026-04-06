# Project 09 — News Category Classification

Multi-class text classification for news articles.

## Model
- TF-IDF Vectorizer
- Logistic Regression (multi-class)

## Evaluation
Macro F1-score used to balance performance across all categories.

## Error Analysis
The confusion matrix highlights overlap between:
- business vs technology news
- health vs lifestyle-related articles

This insight can guide feature engineering or hierarchical classification.

## Artifacts
- Confusion matrix: `reports/figures/confusion_matrix.png`