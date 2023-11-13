import pandas as pd
import torch
from neumf import NeuMFEngine
from torch.utils.data import DataLoader, Dataset
from config import get_configs

# Load Data
print("load data")

data = pd.read_csv("./data/processed/combined1.csv")

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
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Predict on user history
print("predict on user history")

MODEL_STATE = "checkpoints/{}".format("neumf_Epoch9_HR0.0000_NDCG0.0000.model")
config = get_configs(num_users, num_items)["neumf_config"]
engine = NeuMFEngine(config)
engine.model.load_state_dict(torch.load(MODEL_STATE))
engine.model.eval()

with torch.no_grad():
    for userId, itemId, rating in dataloader:
        test_user, test_item = userId, itemId
        test_score = engine.model(test_user, test_item)
        print(
            ("test_user", "test_item", "test_score", "test_score * 5", "rating"),
            (test_user, test_item, test_score, test_score * 5, rating),
            sep="\n",
        )
