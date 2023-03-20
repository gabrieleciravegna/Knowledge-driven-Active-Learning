from typing import List, Tuple

import numpy as np
import torch

from .strategy import Strategy


class RandomSampling(Strategy):

    def loss(self, preds: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        n_sample = preds.shape[0]
        return torch.zeros(n_sample)

    def selection(self, preds: torch.Tensor, labelled_idx: list,
                  n_p: int, *args, **kwargs) -> Tuple[List, torch.Tensor]:

        n_sample = preds.shape[0]
        avail_idx = list(set(np.arange(n_sample)) - set(labelled_idx))
        random_idx: List = np.random.choice(avail_idx, n_p, replace=False).tolist()

        return random_idx, self.loss(preds)
