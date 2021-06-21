import random
import warnings

from torch.utils.data import ConcatDataset
from common import DatasetError

class LimitableDataset:
    def __init__(self, inner, warn=True):
        self._indenes = list(range(len(inner))) #TODO:
        self._yieldable = self._indenes
        self._inner = inner

        self.warn = warn

    def limit(self, m, shuffle=False):
        yieldable = self._indenes[:]

        if shuffle:
            random.shuffle(yieldable)

        if m is not None and len(yieldable) < m:
            msg = {f"Tying to limit a dataset to {m} items,"
                   f"only has {len(yieldable)} in total"}

            if self.warn:
                warnings.warn(msg)
            else:
                raise DatasetError(msg)

        self._yieldable = yieldable[:m]

    def __len__(self):
        return len(self._yieldable)

    def __getitem__(self, idx):
        return self._inner[self._yieldable[idx]]

class LimitedConcatDataset(ConcatDataset):
    def __init__(self, datasets, limit=None, shuffle=False, warn=True):
        self.limit = limit
        self.shuffle = shuffle
        # self.datasets = datasets

        limitables = [LimitableDataset(ds, warn=warn) for ds in datasets]

        for ds in limitables:
            ds.limit(limit, shuffle=shuffle)

        super(LimitedConcatDataset, self).__init__(limitables)

    def shuffle(self):
        for ds in self.datasets:
            ds.limit(self.limit, shuffle=True)
