from typing import List, Tuple, Union

import numpy as np
import torch

from kal.active_strategies.kal import KALSampling
from kal.active_strategies.random import RandomSampling
from kal.active_strategies.strategy import Strategy
from kal.knowledge.expl_to_loss import Expl_2_Loss_CV, Expl_2_Loss
from kal.xai import XAI_TREE


class KALXAISampling(Strategy):
    def __init__(self, rand_points=0, dev=torch.device("cpu"), cv=False, class_names=None,
                 xai_model=XAI_TREE, hidden_size=100, epochs=200, lr=0.001, main_classes=None,
                 mutual_excl=False, double_imp=True, **kwargs):
        super(KALXAISampling, self).__init__()
        assert class_names is not None, "Need to pass the names of the classes/features " \
                                        "for which to extract the explanations"
        self.dropout = False
        self.rand_points = rand_points
        self.xai_model = xai_model(hidden_size, lr, epochs, class_names=class_names, dev=dev)
        self.cv = cv
        self.class_names = class_names
        self.main_classes = main_classes
        self.mutual_excl = mutual_excl
        self.double_imp = double_imp

    def loss(self, preds, *args, formulas=None, uncertainty=False,
             x=None, preds_dropout=None, return_argmax=False, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.cv:
            k_loss = Expl_2_Loss_CV(self.class_names, formulas, uncertainty, self.main_classes,
                                    mutual_excl=self.mutual_excl, double_imp=self.double_imp)
            c_loss, arg_max = k_loss(preds, x=x, return_argmax=True)
        else:
            k_loss = Expl_2_Loss(self.class_names, formulas, uncertainty=uncertainty,
                                 mutual_excl=self.mutual_excl, double_imp=self.double_imp)
            c_loss, arg_max = k_loss(preds, x=x, return_argmax=True)

        if return_argmax:
            return c_loss, arg_max
        return c_loss

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  labels=None, diversity=False, x=None, c_loss=None, arg_max=None,
                  preds_dropout=None, debug=False, **kwargs) -> Tuple[List[np.ndarray], torch.Tensor]:

        n_classes = labels.shape[1] if len(labels.shape) > 1 else 1

        assert (c_loss is not None and arg_max is not None) or \
               (c_loss is None and arg_max is None), \
               "Both c_loss and arg max has to be passed to the KAL selection, or none of them"

        if self.cv:
            formulas = self.xai_model.explain_cv(n_classes, labels, labelled_idx, self.main_classes)
        else:
            formulas = self.xai_model.explain(x, labels, labelled_idx)

        if c_loss is None and arg_max is None:
            c_loss, arg_max = self.loss(preds, x=x, formulas=formulas,
                                        preds_dropout=preds_dropout, return_argmax=True)

        selected_idx, c_loss = KALSampling.selection(self, preds, labelled_idx, n_p, labels=labels,
                                                     diversity=diversity, x=x, c_loss=c_loss, arg_max=arg_max,
                                                     preds_dropout=preds_dropout)

        if self.rand_points > 0:
            rand_idx, rand_loss = RandomSampling().selection(preds, labelled_idx, self.rand_points)
            selected_idx = selected_idx[:-self.rand_points] + rand_idx

        return selected_idx, c_loss


class KALXAIUncSampling(KALXAISampling):
    def loss(self, *args, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        kwargs['uncertainty'] = True
        return super(KALXAIUncSampling, self).loss(*args, **kwargs)


class KALXAIDiversitySampling(KALXAISampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALXAIDiversityUncSampling(KALXAIDiversitySampling):
    def loss(self, *args, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        kwargs['uncertainty'] = True
        return super(KALXAIDiversityUncSampling, self).loss(*args, **kwargs)


class KALXAIDropSampling(KALXAISampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:
        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)


class KALDropUncSampling(KALXAIDropSampling):
    def loss(self, *args, **kwargs) -> torch.Tensor:
        kwargs['uncertainty'] = True
        return super().loss(*args, **kwargs)


class KALXAIDropDiversitySampling(KALXAIDropSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALXAIDropDiversityUncSampling(KALXAIDropSampling):
    def loss(self, *args, **kwargs) -> torch.Tensor:
        kwargs['uncertainty'] = True
        return super().loss(*args, **kwargs)

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)
