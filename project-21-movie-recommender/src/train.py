import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_data

def recommend_movies(user_item, sim_df, target_user, top_k=10):
    watched = user_item.loc[target_user]
    watched_movies = watched[watched > 0].index.tolist()

    similar_users = sim_df[target_user].drop(target_user).sort_values(ascending=False)

    scores = {}
    for other_user, sim_score in similar_users.items():
        for movie, rating in user_item.loc[other_user].items():
            if movie not in watched_movies and rating > 0:
                scores[movie] = scores.get(movie, 0) + sim_score * rating

    recs = (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(top_k)
    )

    return recs

def hit_at_k(recommended_movies, held_out_movie, k=3):
    return int(held_out_movie in recommended_movies.head(k).index.tolist())

def main():
    df = load_data()

    # Hold out one rating from u1 for simple evaluation
    test_row = df[(df["user_id"] == "u1") & (df["movie"] == "Movie_B")].iloc[0]
    train_df = df.drop(test_row.name)

    user_item = train_df.pivot_table(
        index="user_id",
        columns="movie",
        values="rating"
    ).fillna(0)

    sim = cosine_similarity(user_item)
    sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)

    target_user = "u1"
    recs = recommend_movies(user_item, sim_df, target_user, top_k=10)

    hit3 = hit_at_k(recs, test_row["movie"], k=3)

    print(f"Top recommendations for {target_user}:")
    print(recs)
    print(f"\nHit@3: {hit3}")

    Path("reports").mkdir(exist_ok=True)

    recs.reset_index().rename(columns={"index": "movie", 0: "score"}).to_csv(
        "reports/top_recommendations.csv", index=False
    )

    with open("reports/hit_at_k.txt", "w") as f:
        f.write(f"Hit@3: {hit3}\n")

if __name__ == "__main__":
    main()