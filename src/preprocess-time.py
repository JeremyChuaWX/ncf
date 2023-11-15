import pandas as pd

data_dir = "./data/processed/time_weighted_rating_movielens1m.csv"
data = pd.read_csv(data_dir)
print(data.head())
data = data[["userId", "itemId", "timestamp", "time_weighted_rating"]]
data["rating"] = data["time_weighted_rating"]
data = data.drop(labels=["time_weighted_rating"], axis=1)
print(data.head())
data.to_csv("./data/processed/time.csv")
