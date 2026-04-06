import pandas as pd

def create_features(df):
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # Lag features
    df["lag_1"] = df["consumption"].shift(1)
    df["lag_7"] = df["consumption"].shift(7)

    # Rolling stats
    df["roll_mean_7"] = df["consumption"].shift(1).rolling(7).mean()

    return df.dropna()