import math
from typing import List, Tuple, Callable

import numpy as np
import torch

from kal.active_strategies.strategy import Strategy
from kal.knowledge import KnowledgeLoss


class KALSampling(Strategy):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss]):
        super(KALSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=False)

    def loss(self, preds, *args, x=None, return_argmax=False, **kwargs):
        if x is not None:
            c_loss, arg_max = self.k_loss(preds, x=x, return_argmax=True)
        else:
            c_loss, arg_max = self.k_loss(preds, return_argmax=True)

        if return_argmax:
            return c_loss, arg_max

        return c_loss

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int,
                  *args, diversity=False, x=None, c_loss=None, arg_max=None,
                  **kwargs) -> Tuple[List, torch.Tensor]:

        """
        Constrained Active learning strategy.
        We take n elements which are the one that most violates the constraints
        and are among available idx

        :param arg_max:
        :param c_loss:
        :param x:
        :param preds:
        :param labelled_idx: unavailable data (already selected)
        :param n_p: number of points to select
        :param diversity: whether to select points based also on their diversity
        :return list of the selected idx
        """
        assert (c_loss is not None and arg_max is not None) or \
               (c_loss is None and arg_max is None), \
               "Both c_loss and arg max has to be passed to the KAL selection"

        n_sample = preds.shape[0]
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        avail_preds = preds[avail_idx]
        avail_x = x[avail_idx]

        if c_loss is None and arg_max is None:
            c_loss, arg_max = self.loss(avail_preds, x=avail_x, return_argmax=True)

        c_loss = c_loss.clone().detach()
        cal_idx = torch.argsort(c_loss, descending=True)
        cal_idx = cal_idx[:-len(labelled_idx)]

        if diversity:
            # max number of samples per rule 1/2 of the total number of samples
            max_p = math.ceil(n_p / 2)
            selected_idx = []
            arg_loss_dict = {}
            for i, index in enumerate(cal_idx):
                arg_loss = arg_max[index].item()
                if arg_loss in arg_loss_dict:
                    # we allow to break diversity in case we have no samples available
                    if arg_loss_dict[arg_loss] == max_p:
                        continue
                    else:
                        arg_loss_dict[arg_loss] += 1
                else:
                    arg_loss_dict[arg_loss] = 1
                selected_idx.append(index)
                if len(selected_idx) == n_p:
                    break
            if len(selected_idx) < n_p:
                print("Breaking diversity")
                selected_idx = self.selection(preds, labelled_idx, n_p)
            assert len(selected_idx) == n_p, "Error in the diversity " \
                                             "selection operation"
            return selected_idx, c_loss

        return list(cal_idx[:n_p]), c_loss


class KALUncSampling(KALSampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss]):
        super(KALSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)


class KALDropoutSampling(KALSampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:

        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)

    def selection(self, preds: torch.Tensor, labelled_idx: list, *args, preds_dropout=None,
                  diversity=False, x=None, **kwargs) -> Tuple[List, torch.Tensor]:
        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        n_sample = preds.shape[0]
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        avail_preds_dropout = preds_dropout[avail_idx]

        c_loss, arg_max = self.loss(preds, preds_dropout=avail_preds_dropout, return_argmax=True, x=x)

        return super().selection(preds_dropout, *args, c_loss=c_loss, arg_max=arg_max, **kwargs)


class KALDropoutUncSampling(KALDropoutSampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss]):
        super(KALSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)
