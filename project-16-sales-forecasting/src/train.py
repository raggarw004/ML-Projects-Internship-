import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.features import create_features
from pathlib import Path

def generate_data(n_days=365, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days)

    sales = (
        200
        + np.arange(n_days) * 0.3
        + 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        + rng.normal(0, 10, n_days)
    )

    return pd.DataFrame({"date": dates, "sales": sales})

def main(model_type):
    df = generate_data()
    df_feat = create_features(df)

    X = df_feat.drop(columns=["date", "sales"])
    y = df_feat["sales"]

    split = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    print(f"{model_type} MAE:", mae)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, f"models/{model_type}_model.joblib")

    # Plot forecast
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(df_feat["date"].iloc[split:], y_test.values, label="Actual")
    plt.plot(df_feat["date"].iloc[split:], preds, label="Forecast")
    plt.legend()
    plt.title(f"{model_type} Forecast")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{model_type}_forecast.png", dpi=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["linear", "rf"], required=True)
    args = parser.parse_args()

    main(args.model_type)