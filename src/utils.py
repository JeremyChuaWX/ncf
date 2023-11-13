"""
    Some handy functions for pytroch model training ...
"""
import torch


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir):
    state_dict = torch.load(
        model_dir, map_location="cuda"
    )  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


def resume_checkpoint_mps(model, model_dir):
    state_dict = torch.load(
        model_dir, map_location="mps"
    )  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(model):
    assert torch.cuda.is_available(), "CUDA is not available"
    cuda_device = torch.device("cuda")
    model.to(cuda_device)


def use_mps(model):
    assert torch.backends.mps.is_available(), "MPS is not available"
    mps_device = torch.device("mps")
    model.to(mps_device)


def use_optimizer(network, params):
    if params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=params["sgd_lr"],
            momentum=params["sgd_momentum"],
            weight_decay=params["l2_regularization"],
        )
    elif params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=params["adam_lr"],
            weight_decay=params["l2_regularization"],
        )
    elif params["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            network.parameters(),
            lr=params["rmsprop_lr"],
            alpha=params["rmsprop_alpha"],
            momentum=params["rmsprop_momentum"],
        )
    elif params["optimizer"] == "ada":
        optimizer = torch.optim.Adagrad(
            network.parameters(),
            lr=params["ada_lr"],
            weight_decay=params["l2_regularization"],
        )
    return optimizer
