import argparse
import joblib

def main(model_path, resume_text):
    model = joblib.load(model_path)
    pred = model.predict([resume_text])[0]
    print(f"Predicted category: {pred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--resume_text", required=True)
    args = parser.parse_args()

    main(args.model_path, args.resume_text)