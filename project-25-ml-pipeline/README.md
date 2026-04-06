# Project 25 — End-to-End ML Pipeline Project (Capstone-lite)

A professional-style churn prediction pipeline built with scikit-learn.

## Problem
Predict whether a customer is likely to churn based on account and usage features.

## Approach
- Generate synthetic churn data
- Build preprocessing and model steps inside one sklearn Pipeline
- Use numeric and categorical transformations
- Read model parameters from a YAML config file
- Save artifacts and metrics for reproducibility

## Project Structure
- `src/data.py` generates the dataset
- `src/train.py` trains and evaluates the pipeline
- `config/config.yaml` stores parameters
- `models/` stores trained model files
- `reports/` stores metrics and sample predictions

## Results
Main outputs:
- trained pipeline in `models/churn_pipeline.joblib`
- evaluation metrics in `reports/metrics.json`
- example predictions in `reports/sample_predictions.csv`

## Key Insight
A full ML pipeline is more useful than just a trained model because it keeps preprocessing and modeling together, making training and inference cleaner and more reproducible.

## Run
```bash
python -m src.train