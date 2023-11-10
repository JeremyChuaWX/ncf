import torch
from gmf import GMF
from mlp import MLP
from engine import Engine
from utils import use_cuda, resume_checkpoint


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.latent_dim_mf = config["latent_dim_mf"]
        self.latent_dim_mlp = config["latent_dim_mlp"]
        self.latent_dim_cnn = config["latent_dim_cnn"]

        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf
        )
        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf
        )
        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_user_cnn = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim_cnn
        )
        self.embedding_item_cnn = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim_cnn
        )

        self.fc_layers = torch.nn.ModuleList()
        for in_size, out_size in zip(config["layers"][:-1], config["layers"][1:]):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.cnn_layers = torch.nn.ModuleList()
        for in_channel, out_channel in zip(
            config["channels"][:-1], config["channels"][1:]
        ):
            self.cnn_layers.append(
                torch.nn.Conv2d(
                    in_channel, out_channel, kernel_size=3, stride=1, padding=1
                )
            )
            self.cnn_layers.append(torch.nn.BatchNorm2d(out_channel))

        self.affine_output = torch.nn.Linear(
            in_features=config["latent_dim_mf"]
            + config["layers"][-1]
            + (config["channels"][-1] * (self.latent_dim_cnn**2)),
            out_features=1,
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_cnn = self.embedding_user_cnn(user_indices)
        item_embedding_cnn = self.embedding_item_cnn(item_indices)

        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1
        )  # the concat latent vector
        for idx in range(len(self.fc_layers)):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        cnn_matrix = torch.einsum(
            "bi,bj->bij", user_embedding_cnn, item_embedding_cnn
        )  # batch outer product
        cnn_matrix = cnn_matrix.unsqueeze(
            1
        )  # expand to [batch_size, in_channels, height, width]

        for idx in range(len(self.cnn_layers) // 2):
            cnn_matrix = self.cnn_layers[idx](cnn_matrix)  # convolutional layer
            cnn_matrix = self.cnn_layers[idx + 1](cnn_matrix)  # batch normalisation

        cnn_vector = torch.flatten(cnn_matrix, start_dim=1)

        vector = torch.cat([mlp_vector, mf_vector, cnn_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained MLP model & GMF model"""
        config = self.config
        config["latent_dim"] = config["latent_dim_mlp"]
        mlp_model = MLP(config)
        if config["use_cuda"] is True:
            mlp_model.cuda()
        resume_checkpoint(
            mlp_model, model_dir=config["pretrain_mlp"], device_id=config["device_id"]
        )

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        config["latent_dim"] = config["latent_dim_mf"]
        gmf_model = GMF(config)
        if config["use_cuda"] is True:
            gmf_model.cuda()
        resume_checkpoint(
            gmf_model, model_dir=config["pretrain_mf"], device_id=config["device_id"]
        )
        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data],
            dim=-1,
        )
        self.affine_output.bias.data = 0.5 * (
            mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data
        )


class NeuMFEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = NeuMF(config)
        if config["use_cuda"] is True:
            use_cuda(True, config["device_id"])
            self.model.cuda()
        super(NeuMFEngine, self).__init__(config)
        print(self.model)

        if config["pretrain"]:
            self.model.load_pretrain_weights()
