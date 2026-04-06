import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.data import generate_demand
from src.features import add_features

def main():
    df = generate_demand()
    df = add_features(df)

    # Time split (no shuffle)
    split_date = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    X_train = train_df.drop(columns=["demand"])
    y_train = train_df["demand"]
    X_test = test_df.drop(columns=["demand"])
    y_test = test_df["demand"]

    cat_features = ["store_id", "item_id"]
    num_features = ["dow", "month", "lag_1", "lag_7", "roll_mean_7"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    overall_mae = mean_absolute_error(y_test, preds)
    print("Overall MAE:", overall_mae)

    # Top errors by store-item
    eval_df = X_test.copy()
    eval_df["actual"] = y_test.values
    eval_df["pred"] = preds
    eval_df["abs_error"] = (eval_df["actual"] - eval_df["pred"]).abs()

    top_errors = (
        eval_df.groupby(["store_id", "item_id"])["abs_error"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    Path("reports").mkdir(exist_ok=True)
    top_errors.to_csv("reports/top_errors.csv", index=False)

    # Plot forecast for one entity
    sample = eval_df[(eval_df["store_id"] == "store_0") & (eval_df["item_id"] == "item_0")].copy()
    sample = sample.sort_values("date")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(sample["date"], sample["actual"], label="Actual")
    plt.plot(sample["date"], sample["pred"], label="Forecast")
    plt.title("Demand Forecast (store_0, item_0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/forecast_plot.png", dpi=150)

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/demand_forecast_model.joblib")

if __name__ == "__main__":
    main()