import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    df = pd.read_csv("data/sensor_data.csv", parse_dates=["timestamp"])
    model = joblib.load("models/isolation_forest.joblib")

    df["anomaly"] = model.predict(df[["sensor_value"]])
    df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["sensor_value"], label="Sensor Reading")
    plt.scatter(
        df.loc[df["anomaly"] == 1, "timestamp"],
        df.loc[df["anomaly"] == 1, "sensor_value"],
        color="red",
        label="Anomaly"
    )
    plt.legend()
    plt.title("Sensor Anomaly Detection Timeline")
    plt.tight_layout()
    plt.savefig("reports/figures/anomaly_timeline.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()