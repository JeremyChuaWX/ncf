import pandas as pd
import numpy as np

# Load Data
data_dir = f"./data/raw/ratings.dat"
data = pd.read_csv(
    data_dir,
    sep="::",
    header=None,
    names=["uid", "mid", "rating", "timestamp"],
    engine="python",
)

# Reindex
user_id = data[["uid"]].drop_duplicates().reindex()
user_id["userId"] = np.arange(len(user_id))
data = pd.merge(data, user_id, on=["uid"], how="left")

item_id = data[["mid"]].drop_duplicates().reindex()
item_id["itemId"] = np.arange(len(item_id))
data = pd.merge(data, item_id, on=["mid"], how="left")

data = data[["userId", "itemId", "rating", "timestamp"]]

print(f"Range of userId is [{data.userId.min()}, {data.userId.max()}]")
print(f"Unique userIds {data.userId.nunique()}")

print(f"Range of itemId is [{data.itemId.min()}, {data.itemId.max()}]")
print(f"Unique itemIds {data.itemId.nunique()}")

data.to_csv("./data/processed/ratings.csv")
