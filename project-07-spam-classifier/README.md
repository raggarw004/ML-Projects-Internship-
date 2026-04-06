# Project 07 — Spam Email Classifier

Spam detection using TF-IDF and Logistic Regression.

## Model
- TF-IDF Vectorizer
- Linear classifier

## Explainability
The model identifies spam using words strongly associated with promotional language.
Common spam indicators include: "free", "win", "cash", "urgent".

## Example Predictions
```bash
Input: "Win cash prizes now"
Output: Spam

Input: "Can we reschedule the meeting?"
Output: Not Spam