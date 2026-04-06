import argparse
import joblib
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.utils import ensure_dir

def load_data():
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        weights=[0.9, 0.1],  # imbalanced
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)

def main(model_type, out_dir):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    if model_type == "logreg":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )
    elif model_type == "gb":
        model = GradientBoostingClassifier()
    else:
        raise ValueError("Invalid model type")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    ensure_dir(out_dir)
    joblib.dump(pipeline, f"{out_dir}/{model_type}_model.joblib")
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        f"{out_dir}/test_split.joblib"
    )

    print(f"Saved {model_type} model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["logreg", "gb"], required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.model_type, args.out_dir)