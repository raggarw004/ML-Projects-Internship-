import argparse
import joblib

def main(model_path, text):
    model = joblib.load(model_path)
    pred = model.predict([text])[0]
    label = "Positive" if pred == 1 else "Negative"
    print(f"Sentiment: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    main(args.model_path, args.text)