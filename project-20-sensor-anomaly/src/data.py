import numpy as np
import pandas as pd

def generate_sensor_data(n_points=1000, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq="h")

    signal = 50 + rng.normal(0, 2, n_points)

    # Inject anomalies
    anomaly_idx = rng.choice(n_points, size=20, replace=False)
    signal[anomaly_idx] += rng.normal(20, 5, size=len(anomaly_idx))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_value": signal
    })

    return df