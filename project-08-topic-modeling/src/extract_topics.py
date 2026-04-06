import argparse
import joblib
import pandas as pd

def main(model_path, top_n):
    payload = joblib.load(model_path)
    model = payload["model"]
    vectorizer = payload["vectorizer"]

    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for idx, topic in enumerate(model.components_):
        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-top_n - 1:-1]
        ]
        topics.append({
            "topic_id": idx,
            "top_words": ", ".join(top_words)
        })

    df = pd.DataFrame(topics)
    print(df)

    df.to_csv("reports/topics.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--top_n", type=int, default=8)
    args = parser.parse_args()

    main(args.model_path, args.top_n)