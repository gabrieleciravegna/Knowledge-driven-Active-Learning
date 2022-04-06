from typing import Tuple, List

import numpy as np
import torch

from kal.active_strategies.strategy import Strategy


class EntropySampling(Strategy):

    def loss(self, preds, *args, **kwargs) -> torch.Tensor:
        assert len(preds.shape) > 1, "Entropy Sampling requires multi-class prediction"

        log_probs = torch.log(preds)
        uncertainties = (preds * log_probs).sum(1)

        return uncertainties

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int,
                  *args, **kwargs) -> Tuple[List, torch.Tensor]:
        n_sample = preds.shape[0]
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        avail_preds = preds[avail_idx]

        e_loss = self.loss(avail_preds)

        e_idx = torch.argsort(e_loss, descending=True)
        e_idx = e_idx[:n_p].detach().cpu().numpy().tolist()

        return e_idx[:n_p], e_loss


class EntropyDropoutSampling(EntropySampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:
        assert preds_dropout is not None, \
            "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  preds_dropout=None, **kwargs) -> Tuple[List, torch.Tensor]:
        assert preds_dropout is not None, \
            "Need to pass predictions made with dropout to calculate this metric"

        n_sample = preds.shape[0]
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        avail_preds_dropout = preds_dropout[avail_idx]

        e_loss = self.loss(preds, preds_dropout=avail_preds_dropout)

        e_idx = torch.argsort(e_loss, descending=True)
        e_idx = e_idx[:n_p].detach().cpu().numpy().tolist()

        return e_idx, e_loss
