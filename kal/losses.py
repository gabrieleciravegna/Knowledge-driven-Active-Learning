from typing import Callable

import torch
import torch_explain
from torch.nn import NLLLoss

from kal.knowledge import KnowledgeLoss


class CombinedLoss:
    def __init__(self, k_loss: Callable[..., KnowledgeLoss],
                 sup_loss=torch.nn.BCEWithLogitsLoss(),
                 lambda_val: float = 0.5):
        self.k_loss = k_loss(uncertainty=True)
        self.sup_loss = sup_loss
        self.lambda_val = lambda_val
        self.sigmoid = torch.nn.Sigmoid()

    def __call__(self, preds, *args, target=None, **kwargs):
        assert target is not None
        sup_loss = self.sup_loss(preds, target=target)
        preds = self.sigmoid(preds)
        k_loss = self.k_loss(preds, **kwargs)

        return sup_loss + self.lambda_val * k_loss


class EntropyLoss:
    def __init__(self, model, sup_loss=torch.nn.BCEWithLogitsLoss(),
                 lambda_val: float = 0.001):
        self.model = model
        self.sup_loss = sup_loss
        self.lambda_val = lambda_val

    def __call__(self, preds, *args, target=None, **kwargs):
        assert target is not None
        if isinstance(self.sup_loss, NLLLoss) and len(target.shape) > 1:
            target = target.argmax(dim=1)
        sup_loss = self.sup_loss(preds, target=target)
        e_loss = torch_explain.nn.functional.entropy_logic_loss(self.model)
        return sup_loss + self.lambda_val * e_loss
