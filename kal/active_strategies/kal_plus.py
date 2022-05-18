import math
from typing import List, Tuple, Callable, Union

import numpy as np
import sklearn.svm
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from kal.active_strategies.strategy import Strategy
from kal.knowledge import KnowledgeLoss, XORLoss, IrisLoss


class KALPlusSampling(Strategy):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss],
                 detector_model: callable = LocalOutlierFactor,
                 uncertainty=False, **kwargs):
        super(KALPlusSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=uncertainty)
        self.outlier_detector = detector_model()
        self._fitted_detector = False

    def _fit_detector(self, preds_train, *args, x_train=None, **kwargs):
        if isinstance(self.k_loss, IrisLoss) or isinstance(self.k_loss, XORLoss):
            c_losses_train = self.k_loss(preds_train, x=x_train, return_losses=True)
        else:
            c_losses_train = self.k_loss(preds_train, return_losses=True)
        self.outlier_detector.fit(c_losses_train.cpu())
        self._fitted_detector = True

    def loss(self, preds, *args, x=None, return_argmax=None, **kwargs):

        assert self._fitted_detector, "SVM outlier detector needs to be fitted before calling loss"
        if isinstance(self.k_loss, IrisLoss) or isinstance(self.k_loss, XORLoss):
            c_losses_test, arg_max = self.k_loss(preds, x=x, return_argmax=True, return_losses=True)
        else:
            c_losses_test, arg_max = self.k_loss(preds, return_argmax=True, return_losses=True)

        # return negative values for outliers and positive ones for inliers
        self.outlier_detector.novelty = True
        c_losses_plus = self.outlier_detector.decision_function(c_losses_test.cpu())

        # rescaling into (-inf, 0] and then [0, +inf)
        c_losses_plus -= c_losses_plus.max()
        c_losses_plus = torch.as_tensor(np.abs(c_losses_plus))

        if return_argmax:
            return c_losses_plus, arg_max

        return c_losses_plus

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int,
                  *args, x=None, arg_max=None, c_loss_plus=None, diversity=False,
                  preds_dropout=None, debug=False, **kwargs) -> Tuple[List, torch.Tensor]:
        """
        Constrained Active learning strategy.
        We take n elements which are the one that most violates the constraints
        and are among available idx

        :param c_loss_plus:
        :param diversity:
        :param debug:
        :param x:
        :param preds:
        :param labelled_idx: unavailable data (already selected)
        :param n_p: number of points to select
        :return list of the selected idx
        """

        x_train = x[labelled_idx] if x is not None else None
        preds_train = preds[labelled_idx]

        self._fit_detector(preds_train, x_train=x_train)

        assert (c_loss_plus is not None and arg_max is not None) or \
               (c_loss_plus is None and arg_max is None), \
               "Both c_loss_plus and arg max has to be passed to the KAL selection"

        if c_loss_plus is None and arg_max is None:
            c_loss_plus, arg_max = self.loss(preds, x=x, preds_dropout=preds_dropout,
                                             return_argmax=True)

        c_loss_plus[labelled_idx] = -1

        cal_idx = torch.argsort(c_loss_plus, descending=True).tolist()

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
            return selected_idx, c_loss_plus

        selected_idx = cal_idx[:n_p]

        if debug:
            s_loss = torch.nn.CrossEntropyLoss(reduction="none")(preds, kwargs['labels'])
            s_loss[torch.as_tensor(labelled_idx)] = -1
            s_idx = torch.argsort(s_loss, descending=True).cpu().numpy().tolist()[:n_p]
            selected_idx = s_idx[:n_p]
            sns.scatterplot(c_loss_plus, s_loss, style=[2 if idx in selected_idx
                                                        else 1 if idx in labelled_idx else 0
                                                        for idx in range(preds.shape[0])])
            plt.show()

        assert torch.as_tensor([idx not in labelled_idx for idx in selected_idx]).all(), \
            "Error: selected idx already labelled"

        assert len(selected_idx) == n_p, f"Error in selecting the data. " \
                                         f"{len(selected_idx)} points selected instead of {n_p}."

        return selected_idx, c_loss_plus


class KALPlusSamplingSVM(KALPlusSampling):
    def __init__(self, *args, **kwargs):
        if "detector_model" in kwargs:
            kwargs.pop("detector_model")
        super(KALPlusSamplingSVM, self).__init__(*args, **kwargs,
                                                 detector_model=OneClassSVM)


class KALPlusSamplingTree(KALPlusSampling):
    def __init__(self, *args, **kwargs):
        if "detector_model" in kwargs:
            kwargs.pop("detector_model")
        super(KALPlusSamplingTree, self).__init__(*args, **kwargs,
                                                  detector_model=IsolationForest)


class KALPlusSamplingLOF(KALPlusSampling):
    def __init__(self, *args, **kwargs):
        if "detector_model" in kwargs:
            kwargs.pop("detector_model")
        super(KALPlusSamplingLOF, self).__init__(*args, **kwargs,
                                                 detector_model=LocalOutlierFactor)


class KALPlusUncSampling(KALPlusSampling):
    def __init__(self, *args, **kwargs):
        if "uncertainty" in kwargs:
            kwargs.pop("uncertainty")
        super().__init__(*args, uncertainty=True, **kwargs)


class KALPlusDiversitySampling(KALPlusSampling):
    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALPlusUncDiversitySampling(KALPlusSampling):
    def __init__(self, *args, **kwargs):
        if "uncertainty" in kwargs:
            kwargs.pop("uncertainty")
        super().__init__(*args, uncertainty=True, **kwargs)

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALPlusDropSampling(KALPlusSampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:
        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)


class KALPlusDropUncSampling(KALPlusDropSampling):
    def __init__(self, *args, **kwargs):
        if "uncertainty" in kwargs:
            kwargs.pop("uncertainty.py")
        super().__init__(*args, uncertainty=True, **kwargs)


class KALPlusDropDiversitySampling(KALPlusDropSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALPlusDropDiversityUncSampling(KALPlusDropSampling):
    def __init__(self, *args, **kwargs):
        if "uncertainty" in kwargs:
            kwargs.pop("uncertainty.py")
        super().__init__(*args, uncertainty=True, **kwargs)

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)
