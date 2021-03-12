import random

import numpy as np

import torch
import torch.utils.data as data


class DummyDataset(data.Dataset):
    def __init__(self, user_idxs : np.ndarray, item_idxs : np.ndarray, interactions : dict,
        num_items : int):
        assert user_idxs.shape[0] == item_idxs.shape[0]

        self.n_samples = user_idxs.shape[0]

        self.user_idxs = torch.from_numpy(user_idxs)
        self.item_idxs = torch.from_numpy(item_idxs)

        self.interactions = interactions
        self.all = set([i for i in range(num_items)])
   
    def __getitem__(self, index):
        neg_items = list(self.all - set(self.interactions[int(self.user_idxs[index])]))
        neg_idx = random.randint(0, len(neg_items) - 1)
        return self.user_idxs[index], self.item_idxs[index], neg_items[neg_idx]

    def __len__(self):
        return self.n_samples