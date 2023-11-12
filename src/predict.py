import pandas as pd
import random
import torch
from neumf import NeuMFEngine
from torch.utils.data import DataLoader, Dataset

# Load Data
print("load data")

data = pd.read_csv("./data/processed/combined1.csv")

print("Range of userId is [{}, {}]".format(data.userId.min(), data.userId.max()))
print("Range of itemId is [{}, {}]".format(data.itemId.min(), data.itemId.max()))
print("Range of rating is [{}, {}]".format(data.rating.min(), data.rating.max()))

num_users = data.userId.max() + 1
num_items = data.itemId.max() + 1

# Get one user's history
print("get one user history")

user_id = random.randint(0, data.userId.max())
print(f"getting history for user {user_id}")
user_history = data[data["userId"] == user_id].reset_index()

# Initialise dataloader
print("initialise dataloader")


class PredictionDataset(Dataset):
    def __init__(self, user_id, user_history):
        self.userId = user_id

        assert "itemId" in user_history.columns, "itemId column does not exist"
        self.itemId = user_history["itemId"]

        assert "rating" in user_history.columns, "rating column does not exist"
        self.rating = user_history["rating"]

    def __getitem__(self, index):
        return (
            self.userId,
            self.itemId[index],
            self.rating[index],
        )

    def __len__(self):
        return len(self.itemId)


dataset = PredictionDataset(user_id, user_history)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Predict on user history
print("predict on user history")

neumf_config = {
    "alias": "neumf",
    "num_epoch": 10,
    "batch_size": 1024,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim_mf": 8,
    "latent_dim_mlp": 8,
    "latent_dim_cnn": 9,
    "num_negative": 4,
    "layers": [
        16,
        64,
        32,
        16,
        8,
    ],  # layers[0] is the concat of latent user vector & latent item vector
    "channels": [
        1,
        32,
        16,
    ],
    "l2_regularization": 0.01,
    "use_cuda": False,
    "use_mps": False,
    "device_id": 7,
    "pretrain": False,
    "pretrain_mf": "checkpoints/{}".format("gmf_Epoch0_HR0.4756_NDCG0.2175.model"),
    "pretrain_mlp": "checkpoints/{}".format("mlp_Epoch0_HR0.9039_NDCG0.6083.model"),
    "pretrain_cnn": "checkpoints/{}".format("cnn_Epoch0_HR0.1570_NDCG0.0675.model"),
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

MODEL_STATE = "checkpoints/{}".format("neumf_Epoch9_HR0.0000_NDCG0.0000.model")
config = neumf_config
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
