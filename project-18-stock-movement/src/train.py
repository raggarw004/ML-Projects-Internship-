import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.features import create_features
from pathlib import Path

def generate_prices(n_days=600, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days)

    returns = rng.normal(0.0005, 0.01, n_days)
    prices = 100 * (1 + returns).cumprod()

    return pd.DataFrame({"date": dates, "close": prices})

def walk_forward_validation(df, window=200):
    preds, actuals, dates = [], [], []

    for i in range(window, len(df) - 1):
        train = df.iloc[i - window:i]
        test = df.iloc[i:i + 1]

        X_train = train.drop(columns=["date", "close", "target"])
        y_train = train["target"]

        X_test = test.drop(columns=["date", "close", "target"])
        y_test = test["target"]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        preds.append(pred)
        actuals.append(y_test.values[0])
        dates.append(test["date"].values[0])

    return dates, preds, actuals

def main():
    df = generate_prices()
    df_feat = create_features(df)

    dates, preds, actuals = walk_forward_validation(df_feat)

    acc = accuracy_score(actuals, preds)
    print("Walk-forward accuracy:", acc)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(dates, actuals, label="Actual Direction", alpha=0.6)
    plt.plot(dates, preds, label="Predicted Direction", alpha=0.6)
    plt.legend()
    plt.title("Stock Movement Prediction (Walk-Forward)")
    plt.tight_layout()
    plt.savefig("reports/figures/backtest.png", dpi=150)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(acc, "models/walk_forward_accuracy.joblib")

if __name__ == "__main__":
    main()