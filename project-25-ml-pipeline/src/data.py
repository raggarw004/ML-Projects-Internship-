import numpy as np
import pandas as pd

def generate_data(n=500, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n),
        "support_tickets": rng.integers(0, 10, n),
        "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], n),
        "internet_type": rng.choice(["dsl", "fiber", "none"], n),
    })

    churn_score = (
        -0.03 * df["tenure_months"]
        + 0.025 * df["monthly_charges"]
        + 0.35 * df["support_tickets"]
        + (df["contract_type"] == "month-to-month").astype(int) * 1.2
        + (df["internet_type"] == "fiber").astype(int) * 0.4
    )

    prob = 1 / (1 + np.exp(-(churn_score - 1.8)))
    df["churn"] = rng.binomial(1, prob.clip(0, 1))

    return df