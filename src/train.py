import pandas as pd
from config import get_configs
from gmf import GMFEngine
from mlp import MLPEngine
from cnn import CNNEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import argparse

# Parse flags
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="model to train")
args = parser.parse_args()

assert args.model != None, "No model chosen for training"

# Load Data
print("load data")
data = pd.read_csv("./data/processed/ratings.csv")

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
sample_generator = SampleGenerator(data=data)
evaluate_data = sample_generator.evaluate_data

# Specify the exact model
print("start training model")

configs = get_configs(num_users, num_items)

config, engine = None, None
if args.model == "gmf":
    config = configs["gmf_config"]
    engine = GMFEngine(config)
elif args.model == "mlp":
    config = configs["mlp_config"]
    engine = MLPEngine(config)
elif args.model == "cnn":
    config = configs["cnn_config"]
    engine = CNNEngine(config)
elif args.model == "neumf":
    config = configs["neumf_config"]
    engine = NeuMFEngine(config)

assert config != None, "No model chosen for training"
assert engine != None, "No model chosen for training"

for epoch in range(config["num_epoch"]):
    print("Epoch {} starts".format(epoch))
    print("-" * 80)
    train_loader = sample_generator.instance_a_train_loader(config["batch_size"])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    acc = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config["alias"], epoch, acc)
