import pandas as pd
import torch
from gmf import GMFEngine
from mlp import MLPEngine
from cnn import CNNEngine
from neumf import NeuMFEngine
from torch.utils.data import DataLoader, Dataset
from config import get_configs
import argparse

# Parse flags
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="model to predict")
parser.add_argument("--state", default=None, help="model state to load")
args = parser.parse_args()

assert args.model != None, "No model name provided for prediction"
assert args.state != None, "No model file provided for prediction"

MODEL_STATE = "epoch100/{}".format(args.state)

# Load Data
print("load data")

data = pd.read_csv("./data/processed/ratings.csv")

print("Range of userId is [{}, {}]".format(data.userId.min(), data.userId.max()))
print("Range of itemId is [{}, {}]".format(data.itemId.min(), data.itemId.max()))
print("Range of rating is [{}, {}]".format(data.rating.min(), data.rating.max()))

num_users = data.userId.max() + 1
num_items = data.itemId.max() + 1

# Initialise dataloader
print("initialise dataloader")


class PredictionDataset(Dataset):
    def __init__(self, data):
        assert "userId" in data.columns, "userId column does not exist"
        self.userId = data["userId"]

        assert "itemId" in data.columns, "itemId column does not exist"
        self.itemId = data["itemId"]

        assert "rating" in data.columns, "rating column does not exist"
        self.rating = data["rating"]

    def __getitem__(self, index):
        return (
            self.userId[index],
            self.itemId[index],
            self.rating[index],
        )

    def __len__(self):
        return len(self.itemId)


dataset = PredictionDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Predict dataset
print("predict dataset")

predicted_data = data.copy()
predicted_data["predicted"] = 0.0

configs = get_configs(num_users, num_items)

config, engine = None, None

if args.model == "gmf":
    config = configs["gmf_config"]
elif args.model == "mlp":
    config = configs["mlp_config"]
elif args.model == "cnn":
    config = configs["cnn_config"]
elif args.model == "neumf":
    config = configs["neumf_config"]

config["use_cuda"] = False
config["use_mps"] = False
config["pretrain"] = False
config["init"] = False

if args.model == "gmf":
    engine = GMFEngine(config)
elif args.model == "mlp":
    engine = MLPEngine(config)
elif args.model == "cnn":
    engine = CNNEngine(config)
elif args.model == "neumf":
    engine = NeuMFEngine(config)

assert config != None, "No model chosen for prediction"
assert engine != None, "No model chosen for prediction"

engine.model.load_state_dict(torch.load(MODEL_STATE, map_location="cpu"))
engine.model.eval()

with torch.no_grad():
    for idx, (userId, itemId, rating) in enumerate(dataloader):
        test_user, test_item = userId, itemId
        test_score = engine.model(test_user, test_item)
        test_score = test_score.item()
        print(
            test_user.item(),
            test_item.item(),
            test_score * 5,
            rating.item(),
        )
        predicted_data.loc[idx, "predicted"] = test_score * 5

predicted_data["diff"] = predicted_data["rating"] - predicted_data["predicted"]
predicted_data["diff"] = predicted_data["diff"].abs()
predicted_data["acc"] = 1 - (predicted_data["diff"] / predicted_data["rating"])
user_acc = predicted_data.groupby("userId")["acc"].agg("mean")
mean_acc = user_acc.mean()

print("-" * 80)
print(f"model {args.model} with state {args.state}")
print("user accuracy")
print(user_acc)
print("mean accuracy of all users:", mean_acc)

user_acc.to_csv("./data/predict.csv")
