from typing import List, Tuple

import torch

from .strategy import Strategy


class SupervisedSampling(Strategy):
    """
    Supervised Active learning strategy
    Possibly an upper bound to a learning strategy efficacy (fake, obviously).
    We directly select the point which mostly violates the supervision loss.
    """

    def loss(self, preds, labels=None, **kwargs):
        assert labels is not None, "Labels need to be passed in Supervised Sampling loss method"

        s_loss: torch.Tensor = torch.nn.BCELoss(reduction="none")(preds, labels)

        if len(s_loss.shape) > 1:
            s_loss = s_loss.sum(dim=1)
        return s_loss.clone().detach()

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int,
                  *args, labels: torch.Tensor = None, **kwargs) -> Tuple[List, torch.Tensor]:

        """
        :param labels:
        :param preds:
        :param labelled_idx: unavailable data (already selected)
        :param n_p: number of points to select
        :return: selected idx
        """

        assert labels is not None, "Labels are required in the 'fake' supervised strategy"

        s_loss = self.loss(preds, labels)

        s_loss[torch.as_tensor(labelled_idx)] = -1
        sup_idx: List = torch.argsort(s_loss, descending=True).tolist()[:n_p]

        return sup_idx, s_loss
