import math
from typing import List, Tuple, Callable, Union

import torch
import seaborn as sns
from matplotlib import pyplot as plt

from kal.active_strategies.strategy import Strategy
from kal.knowledge import KnowledgeLoss, XORLoss, IrisLoss


class KALSampling(Strategy):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=False)
        self.dropout = False

    def loss(self, preds, *args, x=None, preds_dropout=None, return_argmax=False, **kwargs):
        if isinstance(self.k_loss, IrisLoss) or isinstance(self.k_loss, XORLoss):
            c_loss, arg_max = self.k_loss(preds, x=x, return_argmax=True)
        else:
            c_loss, arg_max = self.k_loss(preds, return_argmax=True)

        if return_argmax:
            return c_loss, arg_max

        return c_loss

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int,
                  *args, diversity=False, x=None, c_loss=None, arg_max=None,
                  preds_dropout=None, debug=False, **kwargs) -> Tuple[List, torch.Tensor]:
        """
        Constrained Active learning strategy.
        We take n elements which are the one that most violates the constraints
        and are among available idx

        :param preds_dropout:
        :param debug:
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

        if c_loss is None and arg_max is None:
            c_loss, arg_max = self.loss(preds, x=x, preds_dropout=preds_dropout,
                                        return_argmax=True)

        c_loss[torch.as_tensor(labelled_idx)] = -1

        cal_idx = torch.argsort(c_loss, descending=True).cpu().numpy().tolist()

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
                # print("Breaking diversity")
                j = 0
                while len(selected_idx) < n_p:
                    if cal_idx[j] not in selected_idx:
                        selected_idx.append(cal_idx[j])
                    j += 1

            assert len(selected_idx) == n_p, "Error in the diversity " \
                                             "selection operation"
            return selected_idx, c_loss

        selected_idx = cal_idx[:n_p]

        if debug:
            s_loss = torch.nn.CrossEntropyLoss(reduction="none")(preds, kwargs['labels'])
            s_loss[torch.as_tensor(labelled_idx)] = -1
            s_idx = torch.argsort(s_loss, descending=True).cpu().numpy().tolist()[:n_p]
            selected_idx = s_idx[:n_p]
            sns.scatterplot(c_loss, s_loss, style=[2 if idx in selected_idx
                                                   else 1 if idx in labelled_idx else 0
                                                   for idx in range(preds.shape[0])])
            plt.show()

        assert torch.as_tensor([idx not in labelled_idx for idx in selected_idx]).all(), \
            "Error: selected idx already labelled"

        assert len(selected_idx) == n_p, f"Error in selecting the data. " \
                                         f"{len(selected_idx)} points selected instead of {n_p}."
        return selected_idx, c_loss


class KALUncSampling(KALSampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)


class KALDiversitySampling(KALSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALDiversityUncSampling(KALDiversitySampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)


class KALDropSampling(KALSampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:
        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)


class KALDropUncSampling(KALDropSampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)


class KALDropDiversitySampling(KALDropSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALDropDiversityUncSampling(KALDropSampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)
