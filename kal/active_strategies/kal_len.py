import math
from typing import List, Tuple, Callable, Union

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset

from kal.active_strategies.strategy import Strategy
from kal.knowledge import KnowledgeLoss
from knowledge.expl_to_loss import Expl_2_Loss_CV, Expl_2_Loss
from network import train_loop, ELEN


class KALLENSampling(Strategy):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], rand_points=0, hidden_size=100,
                 dev=torch.device("cpu"), cv=False, class_names=None, epochs=200, lr= 0.001, **kwargs):
        super(KALLENSampling, self).__init__()
        assert class_names is not None, "Need to pass the names of the classes/features " \
                                        "for which to extract the explanations"
        self.k_loss = k_loss(uncertainty=False)
        self.dropout = False
        self.rand_points = rand_points
        self.hidden_size = hidden_size
        self.dev = dev
        self.cv = cv
        self.class_names = class_names
        self.epochs = epochs
        self.lr = lr

    def loss(self, preds, *args, formulas=None, uncertainty=False,
             x=None, preds_dropout=None, return_argmax=False, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.cv:
            k_loss = Expl_2_Loss_CV(self.class_names, formulas, uncertainty=uncertainty)
            c_loss, arg_max = k_loss(preds, x=x, return_argmax=True)
        else:
            k_loss = Expl_2_Loss(self.class_names, formulas, uncertainty=uncertainty)
            c_loss, arg_max = k_loss(preds, x=x, return_argmax=True)

        if return_argmax:
            return c_loss, arg_max

        return c_loss

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  labels=None, diversity=False, x=None, c_loss=None, arg_max=None,
                  preds_dropout=None, debug=False,**kwargs) -> Tuple[List[np.ndarray], torch.Tensor]:
        n_classes = labels.shape[1]
        expl = [ELEN(input_size=n_classes - 1, hidden_size=self.hidden_size,
                     n_classes=1, dropout=True).to(self.dev) for _ in range(n_classes)]
        formulas = []
        for i in range(n_classes):
            expl_feats = torch.cat([labels[:, :i], labels[:, i + 1:]], dim=1)
            expl_label = labels[:, i].unsqueeze(dim=1)
            expl_dataset = TensorDataset(expl_feats, expl_label)
            train_loop(expl[i], expl_dataset, labelled_idx, self.epochs, lr=self.lr)
            formula = expl[i].explain(expl_feats, expl_label, labelled_idx, self.class_names,
                                      target_class=i)
            formulas.append(formula)

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


class KALLENUncSampling(KALLENSampling):
    def loss(self, *args,  **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        kwargs['uncertainty'] = True
        return super(KALLENUncSampling, self).loss(*args, **kwargs)


class KALLENDiversitySampling(KALLENSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALLENDiversityUncSampling(KALLENDiversitySampling):
    def loss(self, *args,  **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        kwargs['uncertainty'] = True
        return super(KALLENDiversityUncSampling, self).loss(*args, **kwargs)


class KALLENDropSampling(KALLENSampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:
        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)


class KALDropUncSampling(KALLENDropSampling):
    def loss(self, _, *args, **kwargs) -> torch.Tensor:
        kwargs['uncertainty'] = True
        return super().loss(*args, **kwargs)


class KALLENDropDiversitySampling(KALLENDropSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALLENDropDiversityUncSampling(KALLENDropSampling):
    def loss(self, _, *args, **kwargs) -> torch.Tensor:
        kwargs['uncertainty'] = True
        return super().loss(*args, **kwargs)

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)
