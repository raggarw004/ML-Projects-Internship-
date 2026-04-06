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
        "text": [
            "This product is amazing and works perfectly",
            "Terrible experience, completely useless",
            "I love this, would buy again",
            "Worst purchase I have ever made",
            "Very satisfied with the quality",
            "The product broke after one day",
            "Excellent service and great value",
            "Extremely disappointing and frustrating",
            "Happy with the results",
            "Not worth the money at all"
        ],
        "label": [1,0,1,0,1,0,1,0,1,0]
    }
    return pd.DataFrame(data)

def main(out_dir):
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000
        )),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    f1 = f1_score(y_test, preds)

    ensure_dir(out_dir)
    joblib.dump(pipeline, f"{out_dir}/sentiment_pipeline.joblib")

    save_json(
        f"{out_dir}/metrics.json",
        {"f1_score": float(f1)}
    )

    print("F1 Score:", f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.out_dir)