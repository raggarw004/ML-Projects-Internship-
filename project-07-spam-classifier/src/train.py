import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.utils import ensure_dir, save_json

def load_data():
    data = {
        "text": [
            "Congratulations! You have won a free prize",
            "Urgent! Claim your cash reward now",
            "Win money now by clicking this link",
            "Exclusive deal just for you",
            "Hey, are we still meeting tomorrow?",
            "Please find the meeting agenda attached",
            "Can you review this document?",
            "Lunch at 1 pm today?",
            "Limited time offer, buy now",
            "You have been selected for a cash bonus"
        ],
        "label": [1,1,1,1,0,0,0,0,1,1]
    }
    return pd.DataFrame(data)

def main(out_dir):
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=3000
        )),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    f1 = f1_score(y_test, preds)

    ensure_dir(out_dir)
    joblib.dump(pipeline, f"{out_dir}/spam_pipeline.joblib")

    save_json(
        f"{out_dir}/metrics.json",
        {"f1_score": float(f1)}
    )

    # Extract top spam words
    tfidf = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]

    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = model.coef_[0]

    top_spam_idx = np.argsort(coefficients)[-15:]
    top_spam_words = feature_names[top_spam_idx]
    top_spam_scores = coefficients[top_spam_idx]

    ensure_dir("reports/figures")

    plt.figure(figsize=(8, 4))
    plt.barh(top_spam_words, top_spam_scores)
    plt.title("Top Words Indicative of Spam")
    plt.tight_layout()
    plt.savefig("reports/figures/top_spam_words.png", dpi=150)

    print("F1 Score:", f1)
    print("Top spam words:", list(top_spam_words))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.out_dir)