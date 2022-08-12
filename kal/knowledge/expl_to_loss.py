from typing import List, Union, Tuple

import torch
from sympy import to_dnf

from knowledge import KnowledgeLoss


class Expl_2_Loss(KnowledgeLoss):
    def __init__(self, names: List[str], expl: List[str], mu=1, uncertainty=False,
                 mutual_excl=False):
        super().__init__(names)
        self.expl = expl
        self.mu = mu
        self.uncertainty = uncertainty
        self.mutual_excl = mutual_excl

    def __call__(self, output, x=None, targets=False, return_argmax=False, return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Must pass input data to compute " \
                              "the rule violation loss on this dataset"
        if len(output.shape) == 1:
            output = torch.stack([1 - output, output],dim=1)

        from kal.utils import replace_expl_names

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
            loss_fol_product_tnorm.append(impl_loss_1)

            # Computing or_term -> f  = !or_term | f
            # impl_loss_1 = 1 - ((1 - or_term) + f - (1 - or_term) * f)
            # impl_loss_1 = 1 - (1 - or_term - or_term*f)
            # impl_loss_1 = 1 - (1 - or_term* (1 - f))
            impl_loss_2 = or_term * (1 - f)
            loss_fol_product_tnorm.append(impl_loss_2)

        # mutual_excl_loss = 1.0
        # for i in range(output.shape[1]):
        #     excl_loss = output[:, i]
        #     for j in range(output.shape[1]):
        #         if i != j:
        #             excl_loss *= (1 - output[:, j])
        #     mutual_excl_loss *= 1 - excl_loss
        #
        # loss_fol_product_tnorm.append(mutual_excl_loss)

        if self.uncertainty:
            unc_loss = 0
            for i in range(output.shape[1] if len(output.shape) > 1 else 1):
                unc_loss += output[:, i] * (1 - output[:, i])
            loss_fol_product_tnorm.append(unc_loss)

        losses = torch.stack(loss_fol_product_tnorm, dim=1)
        arg_max = torch.argmax(losses, dim=1)

        loss_sum = torch.squeeze(torch.sum(losses, dim=1))

        threshold = 0.5 if targets else 10.
        self.check_loss(output, losses.T, loss_sum, threshold)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum


class Expl_2_Loss_CV(Expl_2_Loss):

    def __call__(self, output, x=None, targets=False, return_argmax=False, return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is None, "Data cannot be used in this loss"
        if len(output.shape) == 1:
            output = torch.stack([1 - output, output],dim=1)

        losses = torch.zeros_like(output[:, :1])
        names = self.names.tolist()
        expl = self.expl
        for i in range(output.shape[1]):
            f = output[:, i]
            x = torch.cat([output[:, :i], output[:, i+1:]], dim=1)
            self.names = names[:i] + names[i+1:]
            self.expl = [expl[i]]
            results = super().__call__(f, x, targets, return_losses=True)
            losses = torch.cat([losses, results], dim=1)
            self.names = names
        self.expl = expl

        loss_sum = torch.squeeze(torch.sum(losses, dim=1))
        arg_max = torch.argmax(losses, dim=1)

        if return_losses:
            if return_argmax:
                return losses, arg_max
            return losses

        if return_argmax:
            return loss_sum, arg_max

        return loss_sum
