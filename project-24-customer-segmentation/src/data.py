import numpy as np
import pandas as pd

def load_data(n=200, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "recency": rng.integers(1, 100, n),
        "frequency": rng.integers(1, 25, n),
        "monetary": rng.integers(20, 1000, n),
    })

    return df