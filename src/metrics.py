import pandas as pd


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores, test_ratings = (
            subjects[0],
            subjects[1],
            subjects[2],
            subjects[3],
        )
        full = pd.DataFrame(
            {
                "user": test_users,
                "item": test_items,
                "score": test_scores,
                "rating": test_ratings,
            }
        )
        self._subjects = full

    def cal_acc(self):
        full = self._subjects
        full["se"] = 0.5 * ((full["rating"] - full["score"]) ** 2)
        return full["se"].mean()
