import torch
from engine import Engine


class CNN(torch.nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.latent_dim_cnn = config["latent_dim_cnn"]

        self.embedding_user_cnn = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim_cnn
        )
        self.embedding_item_cnn = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim_cnn
        )

        self.cnn_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(
            zip(config["cnn_layers"][:-1], config["cnn_layers"][1:])
        ):
            # TODO self.cnn_layers.append(torch.nn.Conv1d(in_size, out_size))
            self.cnn_layers.append(torch.nn.BatchNorm1d(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=config["cnn_layers"][-1],
            out_features=1,
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_cnn = self.embedding_user_cnn(user_indices)
        item_embedding_cnn = self.embedding_item_cnn(item_indices)

        cnn_matrix = torch.outer(user_embedding_cnn, item_embedding_cnn)
        for idx in range(len(self.cnn_layers) // 2):
            cnn_matrix = self.cnn_layers[idx](cnn_matrix)  # convolutional layer
            cnn_matrix = self.cnn_layers[idx + 1](cnn_matrix)  # batch normalisation
        cnn_vector = torch.flatten(cnn_matrix)

        logits = self.affine_output(cnn_vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class CNNEngine(Engine):
    """Engine for training & evaluating CNN model"""

    def __init__(self, config):
        self.model = CNN(config)
        super(CNNEngine, self).__init__(config)
        print(self.model)
