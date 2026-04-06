import numpy as np
import pandas as pd

def load_data(n=200, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "visits": rng.integers(1, 20, n),
        "time_on_site": rng.integers(10, 300, n),
        "email_clicks": rng.integers(0, 10, n),
        "form_submitted": rng.integers(0, 2, n),
    })

    score = (
        0.15 * df["visits"] +
        0.01 * df["time_on_site"] +
        0.4 * df["email_clicks"] +
        1.5 * df["form_submitted"]
    )

    prob = 1 / (1 + np.exp(-(score - 4)))
    df["converted"] = rng.binomial(1, prob.clip(0, 1))

    return df