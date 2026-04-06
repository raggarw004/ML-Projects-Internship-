import argparse
import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error

from src.utils import save_json

def main(model_path, reports_dir):
    model = joblib.load(model_path)
    split = joblib.load("models/test_split.joblib")

    X_test = split["X_test"]
    y_test = split["y_test"]

    preds = model.predict(X_test)
    df_eval = X_test.copy()
    df_eval["actual"] = y_test
    df_eval["predicted"] = preds
    df_eval["abs_error"] = (df_eval["actual"] - df_eval["predicted"]).abs()

    segment_mae = (
        df_eval
        .groupby(["age_band", "bmi_band"])
        .agg(mae=("abs_error", "mean"))
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    save_json(
        f"{reports_dir}/metrics.json",
        {"overall_mae": mean_absolute_error(y_test, preds)}
    )

    segment_mae.to_csv(
        f"{reports_dir}/segment_mae.csv",
        index=False
    )

    print(segment_mae.head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()

    main(args.model_path, args.reports_dir)