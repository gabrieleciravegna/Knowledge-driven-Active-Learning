from typing import Tuple, Union

import torch

from . import KnowledgeLoss


def steep_sigmoid(x: torch.Tensor, k=10., b=0.5) -> torch.Tensor:
    output: torch.Tensor = 1 / (1 + torch.exp(-k * (x - b)))
    # output: torch.Tensor = x > 0.5
    return output


class XORLoss(KnowledgeLoss):
    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    # (x1 & ~x2) | (x2 & ~x1) <=> f
    # => f -> (x1 & ~x2) | (~x1 & x2)
    # => ~f | ((x1 & ~x2) | (~x1 & x2))                            (converting everything into conjunctions)
    # => ~f | (term1 | term2)                                      (each term in CNF is considered a term)
    # => ~f | ~((~term1) & (~term2))
    # => ~((~(~f)) & ~(~(~term1) & (~term2)))
    # => ~(f & ((~term1) & (~term2)))                              (simplifying)
    # => 1 - (f * ((1 - term1) * (1 - term2))                      (converting into product T-norm
    # => (f * ((1 - (x1 * (1 - x2))) * (1 - (x2 * (1 - x1))))      (converting into loss)
    #
    # (x1 & ~x2) | (~x1 & x2) -> f
    # => ~ ((x1 & ~x2) | (~x1 & x2)) | f                           (each term in CNF is considered a term)
    # => ~ (term1 | term2) | f                                     (converting everything into conjunctions)
    # => (~(~((~ term1) and (~ term2))) | f                        (not term1 and not term2) = terms
    # => (~(~(terms)) | f
    # => terms | f
    # => ~(~ terms and ~ f)                                        (simplifying)
    # => 1 - ((1 - terms) * (1 - f))                               (converting into product t-norm)
    # => (1 - terms)) * (1 - f)                                    (converting into loss formulation (1 - formula))
    # => (1 - ((1 - term1) * (1 - term2)) * (1 - f)                (replacing again terms with (1 - term1) * (1 - term2)
    # => 1 - ((1 - (x1 * (1 - x2)) * (1 - (x2 * (1 - x1)) * (1 - f)

    def __call__(self, output: torch.Tensor, return_argmax=False, x: torch.Tensor = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * ((1 - (x1 * (1 - x2))) * (1 - (x2 * (1 - x1))))
        cons_loss2 = (1 - ((1 - (x1 * (1 - x2))) * (1 - (x2 * (1 - x1))))) * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        c_losses = torch.stack(c_losses, dim=1)
        loss_sum = c_losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, c_losses, loss_sum, threshold)

        if return_argmax:
            most_violated_rules = c_losses.argmax(dim=1)
            return loss_sum, most_violated_rules

        return loss_sum
