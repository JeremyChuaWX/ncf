import torch
from engine import Engine
from utils import resume_checkpoint_mps, use_cuda, use_mps, resume_checkpoint


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.latent_dim = config["latent_dim"]

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        self.affine_output = torch.nn.Linear(
            in_features=self.latent_dim, out_features=1
        )

        self.relu = torch.nn.ReLU()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.relu(logits)
        return rating

    def init_weight(self):
        if self.config["use_cuda"]:
            resume_checkpoint(self, model_dir=self.config["init_dir"])
        if self.config["use_mps"]:
            resume_checkpoint_mps(self, model_dir=self.config["init_dir"])


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = GMF(config)
        if config["use_cuda"] is True:
            use_cuda(self.model)
        if config["use_mps"]:
            use_mps(self.model)
        super(GMFEngine, self).__init__(config)
        print(self.model)

        if config["init"]:
            self.model.init_weight()
