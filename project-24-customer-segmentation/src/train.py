import joblib
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.data import load_data

def describe_cluster(row):
    if row["frequency"] >= 15 and row["monetary"] >= 600:
        return "High-value loyal customers"
    elif row["recency"] <= 20 and row["frequency"] >= 8:
        return "Recently active regular customers"
    else:
        return "Lower-engagement customers"

def main():
    df = load_data()

    features = ["recency", "frequency", "monetary"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    df["cluster"] = labels
    sil_score = silhouette_score(X_scaled, labels)

    summary = df.groupby("cluster")[features].mean().reset_index()
    summary["persona"] = summary.apply(describe_cluster, axis=1)

    print("Silhouette score:", sil_score)
    print(summary)

    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df.to_csv("reports/customer_segments.csv", index=False)
    summary.to_csv("reports/cluster_personas.csv", index=False)
    joblib.dump(model, "models/customer_segmentation_model.joblib")

if __name__ == "__main__":
    main()