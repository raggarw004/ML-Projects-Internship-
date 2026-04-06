import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest

from src.data import generate_sensor_data

def main():
    df = generate_sensor_data()

    X = df[["sensor_value"]]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42
    )

    model.fit(X)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/isolation_forest.joblib")

    df.to_csv("data/sensor_data.csv", index=False)
    print("Model trained and data saved.")

if __name__ == "__main__":
    main()