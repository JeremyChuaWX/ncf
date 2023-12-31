import torch
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return (
            self.user_tensor[index],
            self.item_tensor[index],
            self.target_tensor[index],
        )

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, data):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert "userId" in data.columns
        assert "itemId" in data.columns
        assert "rating" in data.columns

        self.data = data

        self.preprocess_data = self._normalize(data)  # explicit feedback
        # self.preprocess_ratings = self._binarize(ratings)  # implicit feedback

        self.train_data, self.test_data = self._split_loo(self.preprocess_data)

    def _normalize(self, data):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        data = deepcopy(data)
        max_rating = data.rating.max()
        data["rating"] = data.rating * 1.0 / max_rating
        return data

    def _binarize(self, data):
        """binarize into 0 or 1, imlicit feedback"""
        data = deepcopy(data)
        data["rating"][data["rating"] > 0] = 1.0
        return data

    def _split_loo(self, data):
        """leave one out train/test split"""
        data["rank_latest"] = data.groupby(["userId"])["timestamp"].rank(
            method="first", ascending=False
        )
        test = data[data["rank_latest"] == 1]
        train = data[data["rank_latest"] > 1]
        assert train["userId"].nunique() == test["userId"].nunique()
        return (
            train[["userId", "itemId", "rating"]],
            test[["userId", "itemId", "rating"]],
        )

    def instance_a_train_loader(self, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []

        for row in self.train_data.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))

        user_tensor = torch.LongTensor(users)
        assert torch.isfinite(user_tensor).all()

        item_tensor = torch.LongTensor(items)
        assert torch.isfinite(item_tensor).all()

        target_tensor = torch.FloatTensor(ratings)
        assert torch.isfinite(target_tensor).all()

        dataset = UserItemRatingDataset(
            user_tensor=user_tensor,
            item_tensor=item_tensor,
            target_tensor=target_tensor,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_users, test_items, test_ratings = [], [], []

        for row in self.test_data.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            test_ratings.append(float(row.rating))

        test_user_tensor = torch.LongTensor(test_users)
        assert torch.isfinite(test_user_tensor).all()

        test_item_tensor = torch.LongTensor(test_items)
        assert torch.isfinite(test_item_tensor).all()

        test_rating_tensor = torch.FloatTensor(test_ratings)
        assert torch.isfinite(test_rating_tensor).all()

        return [test_user_tensor, test_item_tensor, test_rating_tensor]
