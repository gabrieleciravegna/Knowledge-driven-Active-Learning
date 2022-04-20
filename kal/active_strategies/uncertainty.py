from typing import List, Tuple

import torch

from .strategy import Strategy


class UncertaintySampling(Strategy):

    def loss(self, preds: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        We define as uncertainty a metric function for calculating the
        proximity to the boundary (predictions = 0.5).
        In order to be a proper metric function we take the opposite of
        the distance from the boundary mapped into [0,1]
        uncertainty = 1 - 2 * ||preds - 0.5||

        :param preds: predictions of the network
        :return: uncertainty measure
        """
        distance = torch.abs(preds - torch.as_tensor(0.5))
        if len(preds.shape) > 1:
            distance = distance.mean(dim=1)
        uncertainty = 1 - 2 * distance
        return uncertainty

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  u_loss: torch.Tensor = None, **kwargs) \
            -> Tuple[List, torch.Tensor]:

        if u_loss is None:
            u_loss = self.loss(preds)

        u_loss[torch.as_tensor(labelled_idx)] = -1
        u_idx = torch.argsort(u_loss, descending=True).cpu().numpy().tolist()

        return list(u_idx[:n_p]), u_loss


class UncertaintyDropoutSampling(UncertaintySampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:

        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)

    def selection(self, preds, *args, preds_dropout=None, **kwargs) \
            -> Tuple[List, torch.Tensor]:

        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        u_loss = self.loss(preds, *args, preds_dropout=preds_dropout, **kwargs)

        return super().selection(preds_dropout, *args, u_loss=u_loss, **kwargs)
