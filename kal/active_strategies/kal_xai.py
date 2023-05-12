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
                 xai_model=XAI_TREE, hidden_size=100, epochs=200, lr=0.001, height=None,
                 main_classes=None, mutual_excl=False, double_imp=True, discretize_feats=False,
                 attribute_to_classes=False,
                 **kwargs):
        super(KALXAISampling, self).__init__()
        assert class_names is not None, "Need to pass the names of the classes/features " \
                                        "for which to extracts the explanations"
        self.dropout = False
        self.rand_points = rand_points
        self.xai_model = xai_model(hidden_size, epochs, lr, discretize_feats=discretize_feats,
                                   height=height, class_names=class_names, dev=dev)
        self.cv = cv
        self.class_names = class_names
        self.main_classes = main_classes
        self.mutual_excl = mutual_excl
        self.double_imp = double_imp
        self.discretize_feats = discretize_feats
        self.attribute_to_class = attribute_to_classes

    def loss(self, preds, *args, formulas=None, uncertainty=False, labels=None,
             x=None, preds_dropout=None, return_argmax=False, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.cv:
            if len(self.main_classes) == len(self.class_names):
                c_loss, arg_max = [], []
                for i in self.main_classes:
                    attribute_classes = [j for j in self.main_classes if j != i]
                    main_classes = [i]
                    k_loss = Expl_2_Loss_CV(self.class_names, formulas, uncertainty,
                                            main_classes, attribute_classes,
                                            mutual_excl=self.mutual_excl, double_imp=self.double_imp)
                    c_loss += [k_loss(preds)]
                c_loss = torch.stack(c_loss, dim=1)
                arg_max = c_loss.argmax(dim=1)
                c_loss = c_loss.sum(dim=1)
            else:
                attribute_classes = [*range(len(self.main_classes), len(self.class_names))]
                k_loss = Expl_2_Loss_CV(self.class_names, formulas, uncertainty,
                                        self.main_classes, attribute_classes,
                                        mutual_excl=self.mutual_excl, double_imp=self.double_imp,
                                        attribute_to_classes=self.attribute_to_class,)
                c_loss, arg_max = k_loss(preds, return_argmax=True)
        else:
            k_loss = Expl_2_Loss(self.class_names, formulas, uncertainty=uncertainty,
                                 mutual_excl=self.mutual_excl, double_imp=self.double_imp,
                                 discretize_feats=self.discretize_feats)
            x = x.to(preds.device)
            c_loss, arg_max = k_loss(preds, x=x, return_argmax=True)

        if return_argmax:
            return c_loss, arg_max
        return c_loss

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  labels=None, diversity=False, x=None, c_loss=None, arg_max=None, formulas=None,
                  preds_dropout=None, debug=False, **kwargs) -> Tuple[List[np.ndarray], torch.Tensor]:

        n_classes = labels.shape[1] if len(labels.shape) > 1 else 1

        assert (c_loss is not None and arg_max is not None) or \
               (c_loss is None and arg_max is None), \
               "Both c_loss and arg max has to be passed to the KAL selection, or none of them"

        if formulas is None:
            if self.cv:
                if len(self.main_classes) == n_classes:
                    formulas = self.xai_model.explain_cv_multi_class(n_classes, preds, labelled_idx)
                else:
                    formulas = self.xai_model.explain_cv(n_classes, preds, labelled_idx, self.main_classes)
            else:
                formulas = self.xai_model.explain(x, preds, labelled_idx)
        if debug:
            print("Extracted formulas:", formulas)

        if c_loss is None and arg_max is None:
            c_loss, arg_max = self.loss(preds, x=x, formulas=formulas, preds_dropout=preds_dropout,
                                        labels=labels, return_argmax=True, **kwargs)

        selected_idx, c_loss = KALSampling.selection(self, preds, labelled_idx, n_p, labels=labels,
                                                     diversity=diversity, x=x, c_loss=c_loss, arg_max=arg_max,
                                                     preds_dropout=preds_dropout)

        if self.rand_points > 0:
            selected_idx = selected_idx[:-self.rand_points]
            rand_idx, rand_loss = RandomSampling().selection(preds, labelled_idx + selected_idx, self.rand_points)
            selected_idx += rand_idx

            assert len(np.unique(selected_idx)) == n_p, f"Error in selecting the points: {selected_idx}"

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


class KALXAIDropUncSampling(KALXAIDropSampling):
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


