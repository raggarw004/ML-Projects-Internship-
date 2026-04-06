import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from src.data import load_data

def assign_tier(prob):
    if prob >= 0.75:
        return "Hot"
    elif prob >= 0.45:
        return "Warm"
    else:
        return "Cold"

def main():
    df = load_data()

    X = df[["visits", "time_on_site", "email_clicks", "form_submitted"]]
    y = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_model = LogisticRegression(max_iter=1000)
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    results = X_test.copy()
    results["actual"] = y_test.values
    results["conversion_probability"] = probs
    results["lead_tier"] = results["conversion_probability"].apply(assign_tier)

    print("AUC:", auc)
    print(results.head(10))

    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    results.head(20).to_csv("reports/lead_scores.csv", index=False)
    with open("reports/tier_rules.txt", "w") as f:
        f.write("Hot: probability >= 0.75\n")
        f.write("Warm: probability >= 0.45 and < 0.75\n")
        f.write("Cold: probability < 0.45\n")

    joblib.dump(model, "models/lead_scoring_model.joblib")

if __name__ == "__main__":
    main()