from typing import Tuple, Union

import torch
from torch import Tensor

from . import KnowledgeLoss


def steep_sigmoid(x: torch.Tensor, k=10, b=0.5) -> torch.Tensor:
    output: torch.Tensor = 1 / (1 + torch.exp(-k * (x - b)))
    return output


class IrisLoss(KnowledgeLoss):
    def __init__(self, names=None, mu=1,
                 uncertainty: bool = False):
        super().__init__(names)
        self.mu = mu
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, return_argmax=False, x: torch.Tensor = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # ~petal_length <=> f1
        # ~petal_length => f1
        # => ~(~petal_length) | f1
        # => petal_length | f1
        # => ~(~petal_length & ~f1)
        # => 1 - ((1 - petal_length) * (1 - f1))
        # => ((1 - petal_length) * (1 - f1))

        # f1 => ~petal_length
        # => ~petal_length | ~f1
        # => ~(petal_length & f1)
        # => 1 - (petal_length * f1)
        # => (petal_length * f1)

        # petal_length & ~petal_width <=> f2
        # petal_length & ~petal_width => f2
        # ~(petal_length & ~petal_width) | f2
        # ~ ((petal_length & ~petal_width) & ~f2)
        # 1 - ((petal_length * ~petal_width) * (1 - f2))
        # (petal_length * (1 - petal_width) * (1 - f2))

        # petal_length & petal_width <=> f3
        # petal_length & petal_width => f3
        # ~(petal_length & petal_width) | f3
        # ~ ((petal_length & petal_width) & ~f3)
        # 1 - ((petal_length * petal_width) * (1 - f3))
        # (petal_length * petal_width) * (1 - f3)

        # f3 => petal_length & petal_width
        # (petal_length & petal_width) | ~f3
        # ~ (~(petal_length & petal_width) & f3)
        # 1 - ((1 - petal_length * petal_width) * f3)
        # (1 - petal_length * petal_width) * f3

        # f1 =>  ~f2 & ~f3
        # ~f2 & ~f3 | ~f1
        # ~(~(~f2 & ~f3) & ~ ~f1)
        # ~(~(~f2 & ~f3) & f1)
        # 1 - ((1 - ( 1 - f2) * (1 - f3)) * f1
        # (f1 * (1 - (1 - f2) * (1 - f3))

        petal_length = steep_sigmoid(x[:, 2], k=100, b=0.3).float()
        petal_width = steep_sigmoid(x[:, 3], k=100, b=0.6).float()
        f1 = output[:, 0]
        f2 = output[:, 1]
        f3 = output[:, 2]
        c_loss11 = (1 - petal_length) * (1 - f1)
        c_loss12 = (petal_length * f1)
        c_loss21 = (petal_length * (1 - petal_width) * (1 - f2))
        c_loss22 = (1 - petal_length * (1 - petal_width)) * f2
        c_loss31 = (petal_length * petal_width) * (1 - f3)
        c_loss32 = (1 - petal_length * petal_width) * f3
        c_loss4 = (1 - f1 * (1 - f2) * (1 - f3)) * \
                  (1 - f2 * (1 - f1) * (1 - f3)) * \
                  (1 - f3 * (1 - f1) * (1 - f2))
        c_loss_unc1 = f1 * (1 - f1)
        c_loss_unc2 = f2 * (1 - f2)
        c_loss_unc3 = f3 * (1 - f3)
        c_losses = [c_loss11 + c_loss12,
                    c_loss21 + c_loss22,
                    c_loss31 + c_loss32,
                    c_loss4]

        if self.uncertainty:
            c_losses.extend([self.mu * c_loss_unc1,
                             self.mu * c_loss_unc2,
                             self.mu * c_loss_unc3])

        c_losses = torch.stack(c_losses, dim=1)

        loss_sum = c_losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, c_losses, loss_sum, threshold)

        if return_argmax:
            most_violated_rules = c_losses.argmax(dim=1)
            return loss_sum, most_violated_rules

        return loss_sum
