def get_configs(num_users: int, num_items: int):
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
        "pretrain_mf": "checkpoints/{}".format("gmf_Epoch9_HR0.9099_NDCG0.6254.model"),
        "pretrain_mlp": "checkpoints/{}".format("mlp_Epoch9_HR0.9291_NDCG0.6677.model"),
        "pretrain_cnn": "checkpoints/{}".format("cnn_Epoch9_HR0.6630_NDCG0.4633.model"),
        "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
    }

    return {
        "gmf_config": gmf_config,
        "mlp_config": mlp_config,
        "cnn_config": cnn_config,
        "neumf_config": neumf_config,
    }
