from typing import List, Tuple

import numpy as np
import torch


class Strategy:

    def __init__(self, *args, **kwargs):
        pass

    def loss(self, preds, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int,
                  *args, **kwargs) -> Tuple[List[np.ndarray], torch.Tensor]:
        raise NotImplementedError()


