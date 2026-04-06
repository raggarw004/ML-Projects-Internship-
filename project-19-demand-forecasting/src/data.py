import numpy as np
import pandas as pd

def generate_demand(n_days=365, n_items=10, n_stores=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days)

    rows = []
    for store in range(n_stores):
        for item in range(n_items):
            base = rng.uniform(5, 50)
            trend = np.linspace(0, rng.uniform(-5, 5), n_days)
            weekly = rng.uniform(0, 10) * np.sin(2 * np.pi * np.arange(n_days) / 7)
            noise = rng.normal(0, 3, n_days)

            demand = (base + trend + weekly + noise).clip(0)

            for d, y in zip(dates, demand):
                rows.append({
                    "date": d,
                    "store_id": f"store_{store}",
                    "item_id": f"item_{item}",
                    "demand": float(y)
                })

    return pd.DataFrame(rows)