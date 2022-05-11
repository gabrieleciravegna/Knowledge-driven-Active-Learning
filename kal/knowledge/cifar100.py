from typing import Tuple, Union

import numpy as np
import torch

from data.Cifar100 import CLASS_1_HOTS
from . import KnowledgeLoss


class CIFAR100Loss(KnowledgeLoss):
    def __init__(self,
                 names=None, uncertainty=False):

        super().__init__(names)
        self.uncertainty = uncertainty
        self.main_classes = [*range(100)]
        self.attributes = [*range(100, 120)]
        self.combinations = torch.stack([torch.as_tensor(c)[-20:] for c in CLASS_1_HOTS.values()])
        self.mu = 1.

    def __call__(self, output, targets=False, return_argmax=False, return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # Class --> Attributes
        combinations = np.asarray(self.combinations)
        loss_fol_product_tnorm = []
        attribute_outputs = output[:, self.attributes]
        for i in self.main_classes:
            c = output[:, i]
            class_combination = torch.tensor(combinations[i, :], dtype=torch.bool)
            if torch.sum(class_combination.to(torch.int)) > 0:
                output_for_imply = attribute_outputs[:, class_combination]
                loss = c * torch.prod(1 - output_for_imply, dim=1)
                loss_fol_product_tnorm.append(loss)
                assert not targets or loss.sum() == 0, "Error in calculating implications Class -> Attr"

        # Attribute --> Classes
        main_class_outputs = output[:, self.main_classes]
        for j_a, j in enumerate(self.attributes):
            a = output[:, j]
            attribute_combination = torch.tensor(combinations[:, j_a], dtype=torch.bool)
            if torch.sum(attribute_combination.to(torch.int)) > 0:
                output_for_imply = main_class_outputs[:, attribute_combination]
                loss = a * torch.prod(1 - output_for_imply, dim=1)
                loss_fol_product_tnorm.append(loss)
                assert not targets or loss.sum() == 0, "Error in calculating implications Attr -> Class"

        # OR on the main classes
        output_or = (1 - output[:, np.asarray(self.main_classes)])
        loss = self.mu * torch.prod(output_or, dim=1)
        loss_fol_product_tnorm.append(loss)
        assert not targets or loss.sum() == 0, "Error in calculating OR on main classes"

        # OR on the attributes
        output_or = (1 - output[:, np.asarray(self.attributes)])
        loss = self.mu * torch.prod(output_or, dim=1)
        loss_fol_product_tnorm.append(loss)
        assert not targets or loss.sum() == 0, "Error in calculating OR on attributes"

        if self.uncertainty:
            unc_loss = 0
            for i in range(output.shape[1]):
                unc_loss += output[:, i] * (1 - output[:, i])
            loss_fol_product_tnorm.append(unc_loss)

        losses = torch.stack(loss_fol_product_tnorm, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        # losses = torch.sum(losses, dim=1)

        loss_sum = torch.squeeze(torch.sum(losses, dim=1))

        threshold = .5 if targets else 100.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum

