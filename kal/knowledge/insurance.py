from typing import Tuple, Union

import torch

from . import KnowledgeLoss
from ..utils import double_implication_loss, steep_sigmoid, inv_steep_sigmoid


class InsuranceLoss(KnowledgeLoss):
    def __init__(self, names=None, mu=1,
                 uncertainty: bool = False):
        super().__init__(names)
        self.mu = mu
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False, return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Must pass input data to compute " \
                              "the rule violation loss on this dataset"
        # | --- smoker_yes <= 0.50
        # | | --- age <= 42.50
        # | | | --- value: [5416.66]
        # | | --- age > 42.50
        # | | | --- value: [12428.30]
        # | --- smoker_yes > 0.50
        # | | --- bmi <= 29.97
        # | | | --- value: [21503.00]
        # | | --- bmi > 29.97
        # | | | --- value: [41512.02]

        # | --- smoker <= 0.50
        # | | --- age <= 0.53
        # | | | --- value: [0.07]
        # | | --- age > 0.53
        # | | | --- value: [0.18]
        # | --- smoker > 0.50
        # | | --- bmi <= 0.38
        # | | | --- value: [0.33]
        # | | --- bmi > 0.38
        # | | | --- value: [0.64]
        assert len(output.squeeze().shape) == 1, f"Error in given output. I should be (n_sample,), received {output.shape} "
        output = output.squeeze()

        smoker = x[:, 4]
        high_bmi = steep_sigmoid(x[:, 1], b=0.4).float()
        old = steep_sigmoid(x[:, 0], b=0.5).float()
        low_price = inv_steep_sigmoid(output, b=0.1)
        mid_low_price = steep_sigmoid(output, b=0.1) * inv_steep_sigmoid(output, b=0.2)
        mid_high_price = steep_sigmoid(output, b=0.2) * inv_steep_sigmoid(output, b=0.5)
        high_price = steep_sigmoid(output, b=0.5)

        antecedent_low = (1 - smoker) * (1 - old)
        c_loss_low = double_implication_loss(antecedent_low, low_price)
        antecedent_mid_low = (1 - smoker) * old
        c_loss_mid_low = double_implication_loss(antecedent_mid_low, mid_low_price)
        antecedent_mid_high = smoker * (1 - high_bmi)
        c_loss_mid_high = double_implication_loss(antecedent_mid_high, mid_high_price)
        antecedent_high = smoker * high_bmi
        c_loss_high = double_implication_loss(antecedent_high, high_price)

        c_check_loss = 1 - (steep_sigmoid(output, b=-0.1) * inv_steep_sigmoid(output, b=1.1))

        c_losses = [c_loss_low, c_loss_mid_low, c_loss_mid_high, c_loss_high, c_check_loss]

        if self.uncertainty:
            pass

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 10.
        self.check_loss(output, losses.T, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum

    # losses = torch.stack(loss_fol_product_tnorm, dim=0)
    #
    # losses = torch.sum(losses, dim=1)
    #
    # loss_sum = torch.squeeze(torch.sum(losses, dim=0))
    #
    # threshold = 0.5 if targets else 10.
    # self.check_loss(output, losses, loss_sum, threshold)
    #
    # if return_arg_max:
    #     arg_max = torch.argmax(losses, dim=0)
    #     return loss_sum, arg_max
    #
    # return loss_sum
