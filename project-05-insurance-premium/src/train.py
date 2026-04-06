import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import ensure_dir

def create_dataset(n=5000, random_state=42):
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        "age": rng.integers(18, 65, n),
        "bmi": rng.normal(30, 6, n).clip(18, 50),
        "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "dependents": rng.integers(0, 5, n),
        "region": rng.choice(["north", "south", "east", "west"], n),
    })

    base = 2000
    df["premium"] = (
        base
        + df["age"] * 30
        + (df["bmi"] > 30) * 1500
        + (df["smoker"] == "yes") * 8000
        + df["dependents"] * 500
        + rng.normal(0, 1000, n)
    )

    return df

def add_feature_bands(df):
    df = df.copy()
    df["age_band"] = pd.cut(
        df["age"],
        bins=[18, 30, 45, 60, 70],
        labels=["18-30", "31-45", "46-60", "60+"]
    )
    df["bmi_band"] = pd.cut(
        df["bmi"],
        bins=[18, 25, 30, 35, 50],
        labels=["normal", "overweight", "obese", "severely_obese"]
    )
    return df

def main(model_type, out_dir):
    df = create_dataset()
    df = add_feature_bands(df)

    X = df.drop(columns="premium")
    y = df["premium"]

    cat_features = ["smoker", "region", "age_band", "bmi_band"]
    num_features = ["age", "bmi", "dependents"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features)
    ])

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "gb":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("Invalid model type")

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    parser.add_argument("--model_type", choices=["linear", "gb"], required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.model_type, args.out_dir)