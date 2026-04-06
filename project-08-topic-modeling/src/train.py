import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from src.utils import ensure_dir

def load_documents():
    return [
        "AI models are transforming software development and automation",
        "Machine learning improves prediction accuracy in finance",
        "Stock markets react to interest rate changes",
        "Investment banking and financial risk management",
        "Healthy diets and exercise improve mental health",
        "Doctors recommend regular health checkups",
        "Fitness routines improve cardiovascular strength",
        "Football teams prepare for the championship season",
        "The player scored a winning goal in the final match",
        "Basketball and football are popular sports worldwide",
        "Cloud computing enables scalable machine learning systems",
        "Neural networks power modern AI applications",
        "Economic growth depends on fiscal and monetary policy",
        "Hospitals are adopting digital health platforms",
        "Athletes train daily to improve performance"
    ]

def main(model_type, n_topics, out_dir):
    docs = load_documents()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=2000
    )
    X = vectorizer.fit_transform(docs)

    if model_type == "nmf":
        model = NMF(
            n_components=n_topics,
            random_state=42
        )
    elif model_type == "lda":
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42
        )
    else:
        raise ValueError("model_type must be 'nmf' or 'lda'")

    model.fit(X)

    ensure_dir(out_dir)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer},
        f"{out_dir}/{model_type}_topic_model.joblib"
    )

    print(f"Saved {model_type} topic model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["nmf", "lda"], default="nmf")
    parser.add_argument("--n_topics", type=int, default=4)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args.model_type, args.n_topics, args.out_dir)