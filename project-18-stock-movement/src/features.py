import pandas as pd

def create_features(df):
    df = df.copy()

    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()

    df["volatility_5"] = df["return_1"].rolling(5).std()

    df["target"] = (df["return_1"].shift(-1) > 0).astype(int)

    return df.dropna()