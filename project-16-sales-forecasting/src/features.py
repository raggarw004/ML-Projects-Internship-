import pandas as pd

def create_features(df, lags=[1,7,14], windows=[7,14]):
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for w in windows:
        df[f"roll_mean_{w}"] = df["sales"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["sales"].shift(1).rolling(w).std()

    return df.dropna()