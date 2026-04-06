import pandas as pd

import pandas as pd

def add_features(df):
    df = df.sort_values(["store_id", "item_id", "date"]).copy()

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df["lag_1"] = df.groupby(["store_id", "item_id"])["demand"].shift(1)
    df["lag_7"] = df.groupby(["store_id", "item_id"])["demand"].shift(7)

    df["roll_mean_7"] = (
        df.groupby(["store_id", "item_id"])["demand"]
        .transform(lambda s: s.shift(1).rolling(7).mean())
    )

    return df.dropna()
