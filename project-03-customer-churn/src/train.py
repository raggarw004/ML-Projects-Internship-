import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils import ensure_dir

def load_data(n=5000, random_state=42):
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.normal(70, 30, n).clip(20, 150),
        "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], n),
        "internet_service": rng.choice(["dsl", "fiber", "none"], n),
        "payment_method": rng.choice(
            ["credit_card", "bank_transfer", "electronic_check"], n
        ),
    })

    churn_prob = (
        (df["contract_type"] == "month-to-month").astype(int) * 0.35
        + (df["monthly_charges"] > 90).astype(int) * 0.25
        + (df["tenure_months"] < 12).astype(int) * 0.3
    )

    df["churn"] = (rng.random(n) < churn_prob.clip(0, 0.9)).astype(int)

    return df

def main(model_type, out_dir):
    df = load_data()

    X = df.drop(columns="churn")
    y = df["churn"]

    num_features = ["tenure_months", "monthly_charges"]
    cat_features = ["contract_type", "internet_service", "payment_method"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )
    else:
        raise ValueError("Invalid model type")

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

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
    parser.add_argument("--model_type", choices=["logreg", "rf"], required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.model_type, args.out_dir)