import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datafile", default=None, help="datafile to process")
args = parser.parse_args()

assert args.datafile != None, "No model chosen for training"

data_dir = f"./data/raw/{args.datafile}"
filename = (args.datafile.split("."))[0]
data = []

with open(data_dir, "r") as file:
    movie_id = None
    for line in file:
        line = line.strip()
        if line.endswith(":"):
            movie_id = int(line[:-1])
        else:
            customer_id, rating, date = line.split(",")
            data.append([movie_id, int(customer_id), int(rating), date])

data = pd.DataFrame(data, columns=["itemID", "userID", "rating", "timestamp"])

# convert date to timestamp
data["timestamp"] = pd.to_datetime(data["timestamp"]).apply(lambda x: x.timestamp())

# reindex IDs
user_id = data[["userID"]].drop_duplicates().reindex()
user_id["userId"] = np.arange(len(user_id))
data = pd.merge(data, user_id, on=["userID"], how="left")

item_id = data[["itemID"]].drop_duplicates().reindex()
item_id["itemId"] = np.arange(len(item_id))
data = pd.merge(data, item_id, on=["itemID"], how="left")

data = data.drop(columns=["userID", "itemID"])

# TODO remove users with less than 5 interactions

# reduce rows
NUM_USERS = 5000
subset_user_ids = data["userId"].unique()[:NUM_USERS]
data = data[data["userId"].isin(subset_user_ids)]

print(data.head())
print("Range of userId is [{}, {}]".format(data.userId.min(), data.userId.max()))
print("Range of itemId is [{}, {}]".format(data.itemId.min(), data.itemId.max()))
print("Range of rating is [{}, {}]".format(data.rating.min(), data.rating.max()))

data.to_csv(f"./data/processed/{filename}.csv")
