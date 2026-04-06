import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score
)

from src.utils import save_json, ensure_dir

def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.01, 0.9, 200)
    scores = [(t, f1_score(y_true, probs >= t)) for t in thresholds]
    return max(scores, key=lambda x: x[1])

def main(model_path, reports_dir):
    model = joblib.load(model_path)
    split = joblib.load("models/test_split.joblib")

    X_test = split["X_test"]
    y_test = split["y_test"]

    probs = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, probs)
    best_threshold, best_f1 = find_best_threshold(y_test, probs)

    precision, recall, _ = precision_recall_curve(y_test, probs)

    ensure_dir(f"{reports_dir}/figures")

    # PR Curve
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/figures/pr_curve.png", dpi=150)

    metrics = {
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold)
    }

    save_json(f"{reports_dir}/metrics.json", metrics)
    print("Metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()

    main(args.model_path, args.reports_dir)