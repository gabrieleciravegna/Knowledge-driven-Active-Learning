from copy import copy, deepcopy
from typing import List, Union, Tuple, Iterable, Sized

import numpy as np
import torch
from sympy import to_dnf
import matplotlib.pyplot as plt
import seaborn as sns

from kal.knowledge import KnowledgeLoss
from kal.utils import steep_sigmoid, inv_steep_sigmoid, epsilon


class Expl_2_Loss(KnowledgeLoss):
    def __init__(self, names: List[str], expl: List[str], mu=1, uncertainty=False,
                 mutual_excl=False, double_imp=True, percentage=None, discretize_feats=False):
        super().__init__(names)
        self.expl = expl
        self.mu = mu
        self.uncertainty = uncertainty
        self.mutual_excl = mutual_excl
        self.double_imp = double_imp
        self.percentage = percentage
        self.discretize_feats = discretize_feats

    def __call__(self, output, x=None, targets=False, return_argmax=False,
                 return_losses=False, debug=False, biased=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x is not None, "Must pass input data to compute " \
                              "the rule violation loss with this strategy"
        from kal.utils import replace_expl_names

        single_output = False
        if len(output.shape) == 1:
            single_output = True
            output = torch.stack([output, 1 - output], dim=1)
            # output = output.unsqueeze(dim=1)
            assert len(self.expl) == 1

        if self.discretize_feats:
            x = (x > 0.5).to(float)
            output = (output > 0.5).to(float)

        # Calculating the loss associated to the explanation of each class
        loss_fol_product_tnorm = [torch.zeros(output.shape[0], device=output.device)]
        for i, expl_i in enumerate(self.expl):

            f = output[:, i]

            assert len(self.names) == x.shape[1], "names must passed to compute " \
                                                  "the loss from the explanation"

            if expl_i != "":
                or_term = 0.  # False is the neutral term for the OR

                expl_i = replace_expl_names(expl_i, self.names)
                if "<=" in expl_i or ">" in expl_i:
                    diseq = True
                else:
                    diseq = False
                    expl_i = str(to_dnf(expl_i, force=True))
                expl_i = expl_i.replace("(", "").replace(")", "").replace(" ", "")

                for min_term in expl_i.split("|"):
                    # Calculating the loss associated to each minterm
                    term_loss = 1.  # True is the neutral term for the AND
                    for term in min_term.split("&"):
                        neg = False
                        if "~" in term:
                            term = term.replace("~", "")
                            neg = True
                        if term == "True":
                            continue
                        feat_n = int(term[term.index("feature")+7:term.index("feature")+17])
                        feat = x[:, feat_n]
                        if diseq:
                            if "<=" in term:
                                thr = float(term.split("<=")[1])
                                term_loss *= inv_steep_sigmoid(feat, b=thr)
                            else:
                                thr = float(term.split(">")[1])
                                term_loss *= steep_sigmoid(feat, b=thr)
                        else:
                            if neg:
                                term_loss *= inv_steep_sigmoid(feat)
                            else:
                                term_loss *= steep_sigmoid(feat)
                    # The loss associated to the disjunction is the disjunction
                    # of the losses associated to each minterm a | b | c = (a | b) | c
                    or_term = or_term + term_loss - or_term * term_loss

                # Computing or_term -> f = !or_term | f
                # impl_loss_1 = 1 - ((1 - or_term) + f - (1 - or_term) * f)
                # impl_loss_1 = 1 - (1 - or_term + or_term*f)
                # impl_loss_1 = 1 - (1 - or_term * (1 - f))
                impl_loss_1 = or_term * (1 - f)
                # thr = 1e-10
                # impl_loss_1[impl_loss_1 < thr] = 0
                assert not targets or impl_loss_1.sum() == 0, "Error in calculating implications Class <-> Attr"
                loss_fol_product_tnorm.append(impl_loss_1)

                # if biased:
                #     # Computing !or_term -> f = or_term | f
                #     # impl_loss_1 = 1 - (or_term + f - (or_term) * f)
                #     # impl_loss_1 = 1 - (or_term + f - or_term*f)
                #     # impl_loss_1 = 1 - (or_term + f - or_term*f)
                #     impl_loss_biased = 1 - (or_term + f - or_term*f)
                #     loss_fol_product_tnorm.append(impl_loss_biased)

                # Computing f -> or_term = !f | or_term
                # impl_loss_1 = 1 - ((1 - f) + or_term - (1 - f) * or_term)
                # impl_loss_1 = 1 - (1 - f - f*or_term)
                # impl_loss_1 = 1 - (1 - f* (1 - or_term))
                if self.double_imp:
                    impl_loss_2 = f * (1 - or_term)
                    assert not targets or impl_loss_2.sum() == 0, "Error in calculating implications Attr -> Class"
                    loss_fol_product_tnorm.append(impl_loss_2)

        if self.mutual_excl:
            # mutual_excl -> !((f1 & !f2 & ...) | (!f1 & f2 & ...) | ... )
            # mutual_excl -> (!(f1 & !f2 & ...) & !(!f1 & f2 & ...) & ... )
            mutual_excl_loss = torch.ones(output.shape[0], device=output.device)
            for k in range(output.shape[1]):
                excl_loss = deepcopy(output[:, k])
                for j in range(output.shape[1]):
                    if k != j:
                        excl_loss *= (1 - output[:, j])
                mutual_excl_loss *= 1 - excl_loss
            loss_fol_product_tnorm.append(mutual_excl_loss)
            if debug:
                sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=mutual_excl_loss.cpu()).set(title="mu_ex: " + expl_i)
                plt.show()

        losses = torch.stack(loss_fol_product_tnorm, dim=1)

        if self.percentage is not None:
            n_rules = losses.shape[1] * self.percentage // 100
            losses = losses[:, :n_rules]

        if self.uncertainty:
            unc_loss = 0.
            for i in range(output.shape[1]) if not single_output else [0]:
                unc_loss += output[:, i] * (1 - output[:, i])
            losses = torch.cat((losses, unc_loss.unsqueeze(dim=1)), dim=1)

        loss_sum = torch.squeeze(torch.sum(losses, dim=1))

        if debug:
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=f.cpu()).set(title="f: " + expl_i)
            plt.axhline(0.5, 0, 1, c="k")
            plt.axvline(0.5, 0, 1, c="k")
            plt.show()
            # sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=term_loss.cpu()).set(title="term: " + expl_i)
            # plt.show()
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=or_term.cpu()).set(title="or_term: " + expl_i)
            plt.show()
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=impl_loss_1.cpu()).set(title="impl1: " + expl_i)
            plt.show()
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=impl_loss_2.cpu()).set(title="impl2: " + expl_i)
            plt.show()
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=unc_loss.cpu()).set(title="f | !f")
            plt.show()
            sns.scatterplot(x=x[:, 0].cpu(), y=x[:, 1].cpu(), hue=loss_sum.cpu()).set(title="loss sum")
            plt.axhline(0.5, 0, 1, c="k")
            plt.axvline(0.5, 0, 1, c="k")
            plt.show()

        threshold = 0.5 if targets else 10.
        self.check_loss(output, losses.T, loss_sum, threshold)

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
                 attribute_classes: Sized, unc_all=True, mutual_excl=False, double_imp=True, percentage=None,
                 attribute_to_classes=False):
        self.n_classes = len(names)
        self.main_classes = main_classes
        self.attribute_classes = attribute_classes
        super().__init__(names, expl, uncertainty=uncertainty, mutual_excl=mutual_excl,
                         double_imp=double_imp, percentage=percentage)
        self.unc_all = unc_all
        self.attribute_to_classes = attribute_to_classes

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
        class_losses = torch.zeros(output.shape[0], 1)
        if not self.attribute_to_classes:
            self.expl, self.names = expl[self.main_classes], names[self.attribute_classes]
            x = output[:, self.attribute_classes]
            f = output[:, self.main_classes]
            class_losses = super().__call__(f, x, targets, return_losses=True)

        # attribute <-> classes
        attr_losses = torch.zeros(output.shape[0], 1)
        if self.attribute_to_classes or (len(expl) == len(self.main_classes) + len(self.attribute_classes)):
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


if __name__ == "__main__":
    x = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    # OR FORMULA
    formula = ["x1 | x2"]
    output = torch.tensor([
        0.0,
        1.0,
        1.0,
        1.0,
    ])
    loss = Expl_2_Loss(names=["x1", "x2"], expl=formula)
    l = loss(output, x).sum()
    assert np.abs(l - 0) < epsilon, f"Error in computing loss {l:.2f}"

    # WRONG OR FORMULA
    output = torch.tensor([
        0.0,
        0.0,
        1.0,
        0.0,
    ])
    l = loss(output, x).sum()
    assert np.abs(l - 2) < epsilon, f"Error in computing loss {l:.2f}"

    # AND FORMULA
    formula = ["x1 & x2"]
    output = torch.tensor([
        0.0,
        0.0,
        0.0,
        1.0,
    ])
    loss = Expl_2_Loss(names=["x1", "x2"], expl=formula)
    l = loss(output, x).sum()
    assert np.abs(l - 0) < epsilon, f"Error in computing loss {l:.2f}"

    # AND FORMULA WITH NOT
    formula = ["x1 & ~x2"]
    output = torch.tensor([
        0.0,
        0.0,
        1.0,
        0.0,
    ])
    loss = Expl_2_Loss(names=["x1", "x2"], expl=formula)
    l = loss(output, x).sum()
    assert np.abs(l - 0) < epsilon, f"Error in computing loss {l:.2f}"

    output = torch.tensor([
        0.0,
        0.5,
        1.0,
        1.0,
    ])
    loss = Expl_2_Loss(names=["x1", "x2"], expl=[""], mutual_excl=True)
    l = loss(output, x).sum()
    assert np.abs(l - 0.5625) < epsilon, f"Error in computing loss {l:.5f}"

    # AND FORMULA WITH DISEQ
    formula = ["x1 > 0.6 & x2 > 0.3"]
    x = torch.tensor([
        [0.8, 0.5],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.9, 0.7],
    ])
    output = torch.tensor([
        1.0,
        0.0,
        0.0,
        1.0,
    ])
    loss = Expl_2_Loss(names=["x1", "x2"], expl=formula)
    l = loss(output, x).sum()
    assert np.abs(l - 0) < epsilon, f"Error in computing loss {l:.2f}"
