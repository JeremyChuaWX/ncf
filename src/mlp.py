import torch
from engine import Engine
from utils import resume_checkpoint_mps, use_cuda, resume_checkpoint, use_mps


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.latent_dim = config["latent_dim"]

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        self.fc_layers = torch.nn.ModuleList()
        for in_size, out_size in zip(config["layers"][:-1], config["layers"][1:]):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=config["layers"][-1], out_features=1
        )

        self.relu = torch.nn.ReLU()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat(
            [user_embedding, item_embedding], dim=-1
        )  # the concat latent vector

        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        logits = self.affine_output(vector)
        rating = self.relu(logits)
        return rating

    def init_weight(self):
        if self.config["use_cuda"]:
            resume_checkpoint(self, model_dir=self.config["init_dir"])
        if self.config["use_mps"]:
            resume_checkpoint_mps(self, model_dir=self.config["init_dir"])


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = MLP(config)
        if config["use_cuda"] is True:
            use_cuda(self.model)
        if config["use_mps"]:
            use_mps(self.model)
        super(MLPEngine, self).__init__(config)
        print(self.model)

        if config["init"]:
            self.model.init_weight()
