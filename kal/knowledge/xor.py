from typing import Tuple, Union

import torch

from . import KnowledgeLoss
from ..utils import steep_sigmoid


class XORLoss(KnowledgeLoss):
    def __init__(self, uncertainty: bool = False, percentage: bool = None):
        super().__init__()
        self.uncertainty = uncertainty
        if percentage is not None:
            assert percentage in [0, 50, 100], \
                f"Error in the required percentage of knowledge ({percentage} %), available values: 0 %, 50 %, 100 %"
        self.percentage = percentage

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

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False, debug=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Must pass input data to compute " \
                              "the rule violation loss on this dataset"

        if self.percentage is not None:
            if self.percentage == 0:
                return XORLossNone(self.uncertainty)\
                    (output, x, return_argmax, return_losses)
            elif self.percentage == 50:
                return XORLossX1ANDNOTX2(self.uncertainty)\
                    (output, x, return_argmax, return_losses)

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * ((1 - (x1 * (1 - x2))) * (1 - (x2 * (1 - x1))))
        cons_loss2 = (1 - ((1 - (x1 * (1 - x2))) * (1 - (x2 * (1 - x1))))) * (1 - output)
        cons_loss3 = output * (1 - output)
        c_losses = [cons_loss1, cons_loss2]

        if debug:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=output.cpu()).set(title="f")
            plt.show()
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=cons_loss1.cpu()).set(title="x0 & !x1 | x1 & !x0 -> f")
            plt.show()
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=cons_loss2.cpu()).set(title="f -> x0 & !x1 | x1 & !x0")
            plt.show()
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=cons_loss3.cpu()).set(title="f | !f")
            plt.show()

        if self.percentage is not None:
            n_rules = round(len(c_losses) * self.percentage // 100)
            c_losses = c_losses[:n_rules]

        if self.uncertainty:
            c_losses.append(cons_loss3)

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossX1ANDNOTX2(XORLoss):
    """ (x1 & ~x2) <=> f

        f -> (x1 & ~x2)
        f * (1 - (x1 * (1 - x2)))

        (x1 & ~x2) -> f
        (x1 * (1 - x2)) * (1 - f)
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * (1 - (x1 * (1 - x2)))
        cons_loss2 = (x1 * (1 - x2)) * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossNOTX1ANDX2(XORLoss):
    """ (~x1 & x2) <=> f

        f -> (~x1 & x2)
        f * (1 - ((1 - x1) * x2))

        (~x1 & x2) -> f
        ((1 - x1) * x2) * (1 - f)
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * (1 - ((1 - x1) * x2))
        cons_loss2 = ((1 - x1) * x2) * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossX1ANDX2(XORLoss):
    """ (x1 & x2) <=> f

        f -> (x1 & x2)
        f * (1 - (x1 * x2))

        (x1 & x2) -> f
        (x1 * x2) * (1 - f)
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * (1 - (x1 * x2))
        cons_loss2 = (x1 * x2) * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossNOTX1ANDNOTX2(XORLoss):
    """ (~x1 & ~x2) <=> f

        f -> (~x1 & ~x2)
        f * (1 - ((1 - x1) * (1 - x2)))

        (~x1 & ~x2) -> f
        ((1 - x1) * (1 - x2)) * (1 - f)
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * (1 - ((1 - x1) * (1 - x2)))
        cons_loss2 = ((1 - x1) * (1 - x2)) * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossX1(XORLoss):
    """ x1 <=> f

        f -> x1
        f * (1 - x1)

        (x1) -> f
        x1 * (1 - f)
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * (1 - x1)
        cons_loss2 = x1 * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossX2(XORLoss):
    """ x2 <=> f

        f -> x2
        f * (1 - x2)

        (x2) -> f
        x2 * (1 - f)
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * (1 - x2)
        cons_loss2 = x2 * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossNOTX1(XORLoss):
    """ ~x1 <=> f

        f -> ~x1
        f * x1

        ~x1 -> f
        (1 - x1) * (1 - f)
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * x1
        cons_loss2 = (1 - x1) * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossNOTX2(XORLoss):
    """ ~x2 <=> f

        f -> ~x2
        f * x2

        ~x2 -> f
        (1 - x2) * (1 - f)
    """

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        x1 = discrete_x[:, 0]
        x2 = discrete_x[:, 1]
        cons_loss1 = output * x2
        cons_loss2 = (1 - x2) * (1 - output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class XORLossNone(XORLoss):

    def __init__(self, uncertainty: bool = False):
        super().__init__()
        self.uncertainty = uncertainty

    def __call__(self, output: torch.Tensor, x: torch.Tensor = None, return_argmax=False,
                 return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Need to pass input data to compute the violation loss"

        if len(output.shape) > 1:
            output = output[:, 1]
        discrete_x = steep_sigmoid(x).float()
        cons_loss1 = torch.zeros_like(output)
        cons_loss2 = torch.zeros_like(output)
        cons_loss3 = output * (1 - output)

        if self.uncertainty:
            c_losses = [cons_loss1, cons_loss2, cons_loss3]
        else:
            c_losses = [cons_loss1, cons_loss2]

        losses = torch.stack(c_losses, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = losses.sum(dim=1)

        threshold = 1.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


Losses = {
    "": XORLossNone,
    "X1": XORLossX1,
    "X2": XORLossX2,
    "~X1": XORLossNOTX1,
    "~X2": XORLossNOTX2,
    "X1 & ~X2": XORLossX1ANDNOTX2,
    "~X2 & X1": XORLossX1ANDNOTX2,
    "~X1 & X2": XORLossNOTX1ANDX2,
    "X2 & ~X1": XORLossNOTX1ANDX2,
    "X1 & X2": XORLossX1ANDX2,
    "X2 & X1": XORLossX1ANDX2,
    "~X1 & ~X2": XORLossNOTX1ANDNOTX2,
    "(X1 & ~X2) | (~X1 & X2)": XORLoss,
    "(X1 & ~X2) | (X2 & ~X1)": XORLoss,
    "(~X1 & X2) | (X1 & ~X2)": XORLoss,
    "(~X1 & X2) | (~X2 & X1)": XORLoss,
    "(X2 & ~X1) | (~X2 & X1)": XORLoss,
    "(~X2 & X1) | (X2 & ~X1)": XORLoss,
}
