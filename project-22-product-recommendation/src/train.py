import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_data

def get_similar_products(df, sim_matrix, product_name, top_n=3):
    idx = df[df["product_name"] == product_name].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    similar = []
    for i, score in scores[1:top_n+1]:
        similar.append({
            "product_name": df.iloc[i]["product_name"],
            "similarity_score": score
        })

    return pd.DataFrame(similar)

def main():
    df = load_data()

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["description"])

    sim_matrix = cosine_similarity(X)

    query_product = "Running Shoes"
    similar_df = get_similar_products(df, sim_matrix, query_product, top_n=3)

    print(f"Products similar to: {query_product}")
    print(similar_df)

    Path("reports").mkdir(exist_ok=True)
    similar_df.to_csv("reports/similar_products.csv", index=False)

if __name__ == "__main__":
    main()