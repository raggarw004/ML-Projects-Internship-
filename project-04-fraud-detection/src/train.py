import argparse
import joblib
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.utils import ensure_dir

def load_data():
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=6,
        n_redundant=4,
        weights=[0.98, 0.02],
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)

def main(strategy, out_dir):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    if strategy == "class_weight":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )
        X_train_final, y_train_final = X_train, y_train

    elif strategy == "undersample":
        # simple random undersampling
        fraud = y_train == 1
        legit = y_train == 0

        fraud_idx = y_train[fraud].index
        legit_idx = y_train[legit].sample(
            len(fraud_idx) * 5, random_state=42
        ).index

        idx = fraud_idx.union(legit_idx)
        X_train_final = X_train.loc[idx]
        y_train_final = y_train.loc[idx]

        model = LogisticRegression(max_iter=1000)

    else:
        raise ValueError("Invalid strategy")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train_final, y_train_final)

    ensure_dir(out_dir)
    joblib.dump(pipeline, f"{out_dir}/{strategy}_model.joblib")
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        f"{out_dir}/test_split.joblib"
    )

    print(f"Saved model with strategy: {strategy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["class_weight", "undersample"],
        required=True
    )
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.strategy, args.out_dir)