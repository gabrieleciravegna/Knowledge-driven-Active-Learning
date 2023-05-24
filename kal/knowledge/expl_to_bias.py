from typing import Union

import torch
from torch import tensor
import numpy as np

from kal.knowledge.expl_to_loss import Expl_2_Loss


class Expl_2_Bias:
    def __init__(self, k_loss: Expl_2_Loss, target_class: int):
        self.k_loss = k_loss
        self.target_class = target_class
        self.train_bias = None
        self.test_bias = None
        self.normalized = False

    def __call__(self, output, *args, return_tensor=False, debug=False, **kwargs) -> Union[float, tensor]:
        output = (output > 0.5).float()
        valid_output_idx = torch.where(output[:, self.target_class] > 0.5)[0]
        valid_output = output[valid_output_idx]
        assert valid_output[:, self.target_class].mean() >= .5, "Error in output selection"

        b_loss = self.k_loss(valid_output, *args, **kwargs)
        bias_measure = 1 - b_loss

        if debug:
            attr_class = np.argwhere(self.k_loss.names ==
                                     self.k_loss.expl[self.target_class]).item()
            print(torch.hstack((valid_output[:, (attr_class, self.target_class)],
                                bias_measure.unsqueeze(1)))[:10])

        if self.normalized:
            bias_measure = (bias_measure - self.test_bias) / \
                           (self.train_bias - self.test_bias)
        if return_tensor:
            return bias_measure

        return bias_measure.mean()

    def set_normalization_values(self, train_bias: float, test_bias: float):
        assert train_bias > test_bias, "Error in setting bias norm. values: " \
                                       "train bias higher than test bias"

        self.train_bias = train_bias
        self.test_bias = test_bias
        self.normalized = True
