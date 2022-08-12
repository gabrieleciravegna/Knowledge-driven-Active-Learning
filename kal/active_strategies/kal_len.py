import math
from typing import List, Tuple, Callable, Union

import torch
import seaborn as sns
from matplotlib import pyplot as plt

from kal.active_strategies.strategy import Strategy
from kal.knowledge import KnowledgeLoss, XORLoss, IrisLoss


class KALLENSampling(Strategy):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALLENSampling, self).__init__()


        self.k_loss = k_loss(uncertainty=False)
        self.dropout = False


class KALLENUncSampling(KALLENSampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALLENSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)


class KALLENDiversitySampling(KALLENSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALLENDiversityUncSampling(KALDiversitySampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALLENSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)


class KALLENDropSampling(KALLENSampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:
        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)


class KALDropUncSampling(KALLENDropSampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALLENSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)


class KALLENDropDiversitySampling(KALLENDropSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALLENDropDiversityUncSampling(KALLENDropSampling):
    def __init__(self, k_loss: Callable[..., KnowledgeLoss], **kwargs):
        super(KALLENSampling, self).__init__()
        self.k_loss = k_loss(uncertainty=True)

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)
