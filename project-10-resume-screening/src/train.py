import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.utils import ensure_dir, save_json

def load_data():
    data = {
        "resume": [
            "Built scalable APIs using Python and Docker",
            "Developed machine learning models for customer churn",
            "Led product roadmap and cross-functional teams",
            "Designed marketing campaigns and growth strategies",
            "Implemented deep learning models using PyTorch",
            "Analyzed data and built predictive analytics dashboards",
            "Defined product requirements and stakeholder alignment",
            "Managed social media campaigns and SEO optimization",
            "Optimized backend systems and cloud infrastructure",
            "Performed statistical analysis and A/B testing"
        ],
        "role": [
            "software_engineer",
            "data_scientist",
            "product_manager",
            "marketing",
            "data_scientist",
            "data_scientist",
            "product_manager",
            "marketing",
            "software_engineer",
            "data_scientist"
        ]
    }
    return pd.DataFrame(data)

def main(out_dir):
    df = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        df["resume"],
        df["role"],
        test_size=0.4,
        random_state=42,
        stratify=df["role"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000
        )),
        ("model", LogisticRegression(
            max_iter=1000
        ))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")

    ensure_dir(out_dir)
    joblib.dump(pipeline, f"{out_dir}/resume_classifier.joblib")

    save_json(
        f"{out_dir}/metrics.json",
        {"macro_f1": float(f1)}
    )

    print("Macro F1:", f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.out_dir)