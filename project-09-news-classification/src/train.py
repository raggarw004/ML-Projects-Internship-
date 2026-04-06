import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from src.utils import ensure_dir, save_json

def load_data():
    data = {
        "text": [
            "Stock markets rally after positive earnings",
            "Tech companies release new AI software",
            "Football team wins the championship match",
            "Doctors warn about rising health risks",
            "Investment banking sees strong growth",
            "New smartphone launched with advanced features",
            "Basketball player scores record points",
            "Health officials promote regular exercise",
            "Economic outlook improves for global markets",
            "Breakthrough in machine learning research",
            "Tennis finals attract millions of viewers",
            "Nutrition experts recommend balanced diets"
        ],
        "category": [
            "business","technology","sports","health",
            "business","technology","sports","health",
            "business","technology","sports","health"
        ]
    }
    return pd.DataFrame(data)

def main(out_dir):
    df = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"],
        df["category"],
        test_size=0.33,
        random_state=42,
        stratify=df["category"]
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
    joblib.dump(pipeline, f"{out_dir}/news_classifier.joblib")

    save_json(
        f"{out_dir}/metrics.json",
        {"macro_f1": float(f1)}
    )

    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_val, preds, labels=labels)

    ensure_dir("reports/figures")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("News Category Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=150)

    print("Macro F1:", f1)
    print("Confusion Matrix saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.out_dir)