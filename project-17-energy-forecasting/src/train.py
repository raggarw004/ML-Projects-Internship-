import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.features import create_features
from pathlib import Path

def generate_data(n_days=730, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days)

    base = 500
    trend = np.linspace(0, 50, n_days)
    weekly = 40 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    yearly = 60 * np.sin(2 * np.pi * np.arange(n_days) / 365)

    noise = rng.normal(0, 20, n_days)

    consumption = base + trend + weekly + yearly + noise

    return pd.DataFrame({"date": dates, "consumption": consumption})

def main():
    df = generate_data()
    df_feat = create_features(df)

    X = df_feat.drop(columns=["date", "consumption"])
    y = df_feat["consumption"]

    split = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    dates_test = df_feat["date"].iloc[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print("MAE:", mae)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/energy_model.joblib")

    # Error by weekday
    error_df = pd.DataFrame({
        "date": dates_test,
        "error": (y_test.values - preds)
    })
    error_df["weekday"] = error_df["date"].dt.day_name()

    weekday_mae = (
        error_df
        .groupby("weekday")["error"]
        .apply(lambda x: x.abs().mean())
        .reindex([
            "Monday","Tuesday","Wednesday",
            "Thursday","Friday","Saturday","Sunday"
        ])
    )

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4))
    weekday_mae.plot(kind="bar")
    plt.ylabel("MAE")
    plt.title("Forecast Error by Weekday")
    plt.tight_layout()
    plt.savefig("reports/figures/error_by_weekday.png", dpi=150)

if __name__ == "__main__":
    main()