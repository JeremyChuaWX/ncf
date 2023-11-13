import torch
from engine import Engine
from utils import use_cuda, use_mps


class CNN(torch.nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
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

        self.layers = torch.nn.ModuleList()
        for in_channel, out_channel in zip(
            config["channels"][:-1], config["channels"][1:]
        ):
            self.layers.append(
                torch.nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config["padding"],
                )
            )
            self.latent_dim //= config["stride"]

        self.affine_output = torch.nn.Linear(
            in_features=config["channels"][-1] * (self.latent_dim**2),
            out_features=1,
        )

        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        matrix = torch.einsum(
            "bi,bj->bij", user_embedding, item_embedding
        )  # batch outer product
        matrix = matrix.unsqueeze(
            1
        )  # expand to [batch_size, in_channels, height, width]

        for idx in range(len(self.layers)):
            matrix = self.layers[idx](matrix)
            matrix = torch.nn.ReLU()(matrix)

        vector = torch.flatten(matrix, start_dim=1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class CNNEngine(Engine):
    """Engine for training & evaluating CNN model"""

    def __init__(self, config):
        self.model = CNN(config)
        if config["use_cuda"] is True:
            use_cuda(self.model)
        if config["use_mps"]:
            use_mps(self.model)
        super(CNNEngine, self).__init__(config)
        print(self.model)
