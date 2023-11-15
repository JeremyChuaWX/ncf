import torch
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model!
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(
            log_dir="runs/{}".format(config["alias"])
        )  # tensorboard writer
        self._writer.add_text("config", str(config), 0)
        self.opt = use_optimizer(self.model, config)

        self.crit = torch.nn.MSELoss()  # explicit feedback
        # self.crit = torch.nn.BCELoss()  # implicit feedback

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, "model"), "Please specify the exact model!"

        if self.config["use_cuda"]:
            users, items, ratings = (
                users.to("cuda"),
                items.to("cuda"),
                ratings.to("cuda"),
            )
        if self.config["use_mps"]:
            users, items, ratings = users.to("mps"), items.to("mps"), ratings.to("mps")

        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print(
                "[Training Epoch {}] Batch {}, Loss {}".format(epoch_id, batch_id, loss)
            )
            total_loss += loss
        self._writer.add_scalar("model/loss", total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, "model"), "Please specify the exact model!"
        self.model.eval()
        with torch.no_grad():
            test_users, test_items, test_ratings = (
                evaluate_data[0],
                evaluate_data[1],
                evaluate_data[2],
            )

            if self.config["use_cuda"]:
                test_users = test_users.to("cuda")
                test_items = test_items.to("cuda")
                test_ratings = test_ratings.to("cuda")

            if self.config["use_mps"]:
                test_users = test_users.to("mps")
                test_items = test_items.to("mps")
                test_ratings = test_ratings.to("mps")

            test_scores = self.model(test_users, test_items)

            if self.config["use_cuda"] or self.config["use_mps"]:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()

            self._metron.subjects = [
                test_users.data.view(-1).tolist(),
                test_items.data.view(-1).tolist(),
                test_scores.data.view(-1).tolist(),
                test_ratings.data.view(-1).tolist(),
            ]

        acc = self._metron.cal_acc()
        self._writer.add_scalar("performance/ACC", acc, epoch_id)
        print("[Evluating Epoch {}] ACC = {:.4f}".format(epoch_id, acc))
        return acc

    def save(self, alias, epoch_id, acc):
        assert hasattr(self, "model"), "Please specify the exact model!"
        model_dir = self.config["model_dir"].format(alias, epoch_id, acc)
        save_checkpoint(self.model, model_dir)
