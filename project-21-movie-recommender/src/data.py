import pandas as pd

def load_data():
    data = [
        ["u1", "Movie_A", 5],
        ["u1", "Movie_B", 4],
        ["u1", "Movie_C", 1],
        ["u2", "Movie_A", 5],
        ["u2", "Movie_B", 5],
        ["u2", "Movie_D", 4],
        ["u3", "Movie_A", 2],
        ["u3", "Movie_C", 5],
        ["u3", "Movie_D", 4],
        ["u4", "Movie_B", 4],
        ["u4", "Movie_C", 4],
        ["u4", "Movie_E", 5],
        ["u5", "Movie_A", 4],
        ["u5", "Movie_D", 5],
        ["u5", "Movie_E", 4],
    ]

    return pd.DataFrame(data, columns=["user_id", "movie", "rating"])