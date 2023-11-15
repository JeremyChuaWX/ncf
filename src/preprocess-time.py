import pandas as pd

data_dir = "./data/processed/time_weighted_rating_movielens1m.csv"
data = pd.read_csv(data_dir)
data = data[["user_id", "itemId", "time_weighted_ratings"]]
data["ratings"] = data["time_weighted_ratings"]
data.to_csv("./data/processed/time.csv")
