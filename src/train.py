import pandas as pd
from gmf import GMFEngine
from mlp import MLPEngine
from cnn import CNNEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import argparse

# Parse flags
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="model to train")
parser.add_argument("--data", default=None, help="data to train on")
args = parser.parse_args()

assert args.model != None, "No model chosen for training"
assert args.data != None, "No data provided for training"

# Load Data
print("load data")
data = pd.read_csv(f"./data/processed/{args.data}")

print(f"Range of userId is [{data.userId.min()}, {data.userId.max()}]")
print(f"Unique userIds {data.userId.nunique()}")

print(f"Range of itemId is [{data.itemId.min()}, {data.itemId.max()}]")
print(f"Unique itemIds {data.itemId.nunique()}")

print(f"Range of rating is [{data.rating.min()}, {data.rating.max()}]")
print(f"Unique ratings {data.rating.nunique()}")

num_users = data.userId.max() + 1
num_items = data.itemId.max() + 1

# DataLoader for training
print("initialise sample generator")
sample_generator = SampleGenerator(ratings=data)
evaluate_data = sample_generator.evaluate_data

# Specify the exact model
print("start traing model")

gmf_config = {
    "alias": "gmf",
    "num_epoch": 10,
    "batch_size": 1024,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim": 8,
    "num_negative": 4,
    "l2_regularization": 0,  # 0.01
    "use_cuda": False,
    "use_mps": True,
    "device_id": 0,
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

mlp_config = {
    "alias": "mlp",
    "num_epoch": 10,
    "batch_size": 1024,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim": 32,
    "num_negative": 4,
    "layers": [
        64,
        32,
        16,
        8,
    ],  # layers[0] is the concat of latent user vector & latent item vector
    "l2_regularization": 0.0000001,  # MLP model is sensitive to hyper params
    "use_cuda": False,
    "use_mps": True,
    "device_id": 7,
    "pretrain": False,
    "pretrain_mf": "checkpoints/{}".format(
        "gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model"
    ),
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

cnn_config = {
    "alias": "cnn",
    "num_epoch": 10,
    "batch_size": 1024,
    "optimizer": "ada",
    "ada_lr": 1e-3,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim": 32,
    "num_negative": 4,
    "channels": [
        1,
        32,
        32,
        32,
        32,
        32,
    ],
    "stride": 2,
    "kernel_size": 2,
    "padding": 0,
    "l2_regularization": 0.0000001,  # CNN model is sensitive to hyper params
    "use_cuda": False,
    "use_mps": True,
    "device_id": 7,
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

neumf_config = {
    "alias": "neumf",
    "num_epoch": 10,
    "batch_size": 1024,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim_mf": gmf_config["latent_dim"],
    "latent_dim_mlp": mlp_config["latent_dim"],
    "latent_dim_cnn": cnn_config["latent_dim"],
    "num_negative": 4,
    "layers": mlp_config["layers"],
    "channels": cnn_config["channels"],
    "stride": cnn_config["stride"],
    "kernel_size": cnn_config["kernel_size"],
    "padding": cnn_config["padding"],
    "l2_regularization": 0.01,
    "use_cuda": False,
    "use_mps": True,
    "device_id": 7,
    "pretrain": True,
    "pretrain_mf": "checkpoints/{}".format("gmf_Epoch9_HR0.8815_NDCG0.6105.model"),
    "pretrain_mlp": "checkpoints/{}".format("mlp_Epoch9_HR0.9257_NDCG0.6652.model"),
    "pretrain_cnn": "checkpoints/{}".format("cnn_Epoch9_HR0.7855_NDCG0.5875.model"),
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

config, engine = None, None
if args.model == "gmf":
    config = gmf_config
    engine = GMFEngine(config)
elif args.model == "mlp":
    config = mlp_config
    engine = MLPEngine(config)
elif args.model == "cnn":
    config = cnn_config
    engine = CNNEngine(config)
elif args.model == "neumf":
    config = neumf_config
    engine = NeuMFEngine(config)

assert config != None, "No model chosen for training"
assert engine != None, "No model chosen for training"

for epoch in range(config["num_epoch"]):
    print("Epoch {} starts".format(epoch))
    print("-" * 80)
    train_loader = sample_generator.instance_a_train_loader(
        config["num_negative"], config["batch_size"]
    )
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)

    if epoch == config["num_epoch"] - 1:
        engine.save(config["alias"], epoch, hit_ratio, ndcg)
