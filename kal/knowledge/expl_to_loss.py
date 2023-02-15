from copy import copy, deepcopy
from typing import List, Union, Tuple, Iterable, Sized

import numpy as np
import torch
from sympy import to_dnf

from kal.knowledge import KnowledgeLoss


class Expl_2_Loss(KnowledgeLoss):
    def __init__(self, names: List[str], expl: List[str], mu=1, uncertainty=False,
                 mutual_excl=False, double_imp=True, percentage=None):
        super().__init__(names)
        self.expl = expl
        self.mu = mu
        self.uncertainty = uncertainty
        self.mutual_excl = mutual_excl
        self.double_imp = double_imp
        self.percentage = percentage

    def __call__(self, output, x=None, targets=False, return_argmax=False, return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Must pass input data to compute " \
                              "the rule violation loss with this strategy"
        from kal.utils import replace_expl_names

        if len(output.shape) == 1:
            output = torch.stack([output, 1 - output], dim=1)
            # output = output.unsqueeze(dim=1)
            assert len(self.expl) == 1

        # Calculating the loss associated to the explanation of each class
        loss_fol_product_tnorm = []
        for i, expl_i in enumerate(self.expl):
            if expl_i == "":
                continue

            f = output[:, i]

            assert len(self.names) == x.shape[1], "names must passed to compute " \
                                                  "the loss from the explanation"

            expl_i = replace_expl_names(expl_i, self.names)
            expl_i = str(to_dnf(expl_i, force=True))
            expl_i = expl_i.replace("(", "").replace(")", "").replace(" ", "")

            or_term = 0.  # False is the neutral term for the OR
            # if len(expl_i.split("|")) != 1:
            #     print(f"Warning: having {len(expl_i.split('|'))} or in the extracted rule")

            for min_term in expl_i.split("|"):
                # Calculating the loss associated to each minterm
                term_loss = 1.  # True is the neutral term for the AND
                for term in min_term.split("&"):
                    neg = False
                    if "~" in term:
                        term = term.replace("~", "")
                        neg = True
                    feat_n = int(term.replace("feature", ""))
                    feat = x[:, feat_n]
                    if neg:
                        term_loss *= (1 - feat)
                    else:
                        term_loss *= feat
                # The loss associated to the disjunction is the disjunction
                # of the losses associated to each minterm a | b | c = (a | b) | c
                or_term = or_term + term_loss - or_term * term_loss

            # Computing f -> or_term = !f | or_term
            # impl_loss_1 = 1 - ((1 - f) + or_term - (1 - f) * or_term)
            # impl_loss_1 = 1 - (1 - f - f*or_term)
            # impl_loss_1 = 1 - (1 - f* (1 - or_term))
            impl_loss_1 = f * (1 - or_term)
            assert not targets or impl_loss_1.sum() == 0, "Error in calculating implications Class <-> Attr"
            loss_fol_product_tnorm.append(impl_loss_1)

            # Computing or_term -> f  = !or_term | f
            # impl_loss_1 = 1 - ((1 - or_term) + f - (1 - or_term) * f)
            # impl_loss_1 = 1 - (1 - or_term + or_term*f)
            # impl_loss_1 = 1 - (1 - or_term * (1 - f))
            if self.double_imp:
                impl_loss_2 = or_term * (1 - f)
                assert not targets or impl_loss_2.sum() == 0, "Error in calculating implications Attr -> Class"
                loss_fol_product_tnorm.append(impl_loss_2)

        if self.mutual_excl:
            mutual_excl_loss = torch.ones(output.shape[0], device=output.device)
            for k in range(output.shape[1]):
                excl_loss = deepcopy(output[:, k])
                for j in range(output.shape[1]):
                    if k != j:
                        excl_loss *= (1 - output[:, j])
                mutual_excl_loss *= 1 - excl_loss
            loss_fol_product_tnorm.append(mutual_excl_loss)

        losses = torch.stack(loss_fol_product_tnorm, dim=1)

        if self.percentage is not None:
            n_rules = losses.shape[1] * self.percentage // 100
            losses = losses[:, :n_rules]

        if self.uncertainty:
            unc_loss = 0.
            for i in range(output.shape[1]):
                unc_loss += output[:, i] * (1 - output[:, i])
            losses = torch.cat((losses, unc_loss.unsqueeze(dim=1)), dim=1)

        loss_sum = torch.squeeze(torch.sum(losses, dim=1))

        threshold = 0.5 if targets else 10.
        # self.check_loss(output, losses.T, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                arg_max = torch.argmax(losses, dim=1)
                return losses, arg_max
            return losses

        if return_argmax:
            arg_max = torch.argmax(losses, dim=1)
            return loss_sum, arg_max

        return loss_sum


class Expl_2_Loss_CV(Expl_2_Loss):

    def __init__(self, names: List[str], expl: List[str], uncertainty: bool, main_classes: Sized,
                 unc_all=True, mutual_excl=False, double_imp=True, percentage=None):
        self.n_classes = len(names)
        self.main_classes = main_classes
        self.attribute_classes = range(len(self.main_classes), self.n_classes)
        super().__init__(names, expl, uncertainty=uncertainty, mutual_excl=mutual_excl,
                         double_imp=double_imp, percentage=percentage)
        self.unc_all = unc_all

    def __call__(self, output, x=None, targets=False, return_argmax=False, return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is None, "Data cannot be used in this loss"
        if len(output.shape) == 1:
            output = torch.stack([1 - output, output], dim=1)

        uncertainty = self.uncertainty
        expl = np.asarray(self.expl)
        names = np.asarray(self.names)
        self.uncertainty = False

        # class <-> attributes
        self.expl, self.names = expl[self.main_classes], names[self.attribute_classes]
        x = output[:, self.attribute_classes]
        f = output[:, self.main_classes]
        class_losses = super().__call__(f, x, targets, return_losses=True)

        # attribute <-> classes
        attr_losses = torch.zeros_like(class_losses)
        if len(self.expl) == len(self.main_classes) + len(self.attribute_classes):
            self.expl, self.names = expl[self.attribute_classes], names[self.main_classes]
            x = output[:, self.main_classes]
            f = output[:, self.attribute_classes]
            attr_losses = super().__call__(f, x, targets, return_losses=True)

        losses = torch.cat([class_losses, attr_losses], dim=1)
        self.uncertainty = uncertainty
        self.expl = expl
        self.names = names

        # OR on the main classes
        if self.percentage is None and self.unc_all:
            output_or = (1 - f)
            or_loss = torch.prod(output_or, dim=1)
            losses = torch.cat((losses, or_loss.unsqueeze(dim=1)), dim=1)

        # OR on the attributes
        if self.percentage is None and self.unc_all:
            output_or = (1 - x)
            or_loss = torch.prod(output_or, dim=1)
            losses = torch.cat((losses, or_loss.unsqueeze(dim=1)), dim=1)

        if self.uncertainty:
            unc_loss = 0.
            for i in range(output.shape[1]):
                unc_loss += output[:, i] * (1 - output[:, i])
            losses = torch.cat((losses, unc_loss.unsqueeze(dim=1)), dim=1)

        loss_sum = torch.squeeze(torch.sum(losses, dim=1))
        arg_max = torch.argmax(losses, dim=1)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


# if __name__ == "__main__":
#     output = torch.tensor([
#         [0., 1., 0., 0.],
#         [0.1, 0.6, 0.4, 0.0],
#         [0., 0., 0., 0.1],
#         [1.0, 0.9, 0.0, 0.0],
#         [1.0, 1.0, 0.1, 0.2],
#         [1.0, 1.0, 0., 0.]
#     ])
#
#     mutual_excl_loss = torch.ones(output.shape[0])
#     for it in range(output.shape[1]):
#         excl_loss = deepcopy(output[:, it])
#         for j in range(output.shape[1]):
#             if it != j:
#                 excl_loss *= (1 - output[:, j])
#         mutual_excl_loss *= 1 - excl_loss
#
#     print(mutual_excl_loss)
