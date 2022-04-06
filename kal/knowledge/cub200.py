from typing import Tuple, Union

import numpy as np
import torch

from . import KnowledgeLoss


class CUB200Loss(KnowledgeLoss):
    def __init__(self, main_classes: list, attributes: list, combinations: list,
                 names=None, scale="none", uncertainty=False, mu=10.0):

        super().__init__(names)
        self.scale = scale
        self.uncertainty = uncertainty
        self.mu = mu
        self.main_classes = main_classes
        self.attributes = attributes
        self.combinations = combinations

    def __call__(self, output, targets=False, return_arg_max=False) \
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

        losses = torch.stack(loss_fol_product_tnorm, dim=0)

        if self.scale:
            if self.scale == "a" or self.scale == "both":
                # scale the first group of rules for the number of predictions made
                # (they may become noisy)
                num_preds = (output > 0.5).sum(dim=1)
                scaling = torch.ones(output.shape[0]) / (num_preds + 1)  # to avoid numerical problem
                scaled_losses = losses[:44] * scaling
                losses[:44] = scaled_losses
            if self.scale == "c" or self.scale == "both":
                # scale by a factor 10 the penultimate rule (which is the most important)
                losses[-2] = losses[-2] * self.mu

        losses = torch.sum(losses, dim=1)

        loss_sum = torch.squeeze(torch.sum(losses, dim=0))

        threshold = .5 if targets else 100.
        self.check_loss(output, losses, loss_sum, threshold)

        if return_arg_max:
            arg_max = torch.argmax(losses, dim=0)
            return loss_sum, arg_max
        return loss_sum

#
# def write_cub_rules(class_names: List[str], main_classes: list, attributes: list,
#                     combinations: list):
#     # Class --> Attributes
#     combinations = np.asarray(combinations)
#     rules = []
#     attribute_outputs = [class_names[i] for i in attributes]
#     for i in main_classes:
#         c = class_names[i]
#         rule = c[:-1] + " -> "
#         class_combination = torch.tensor(combinations[i, :], dtype=torch.bool)
#         if torch.sum(class_combination.to(torch.int)) > 0:
#             class_for_imply = [attribute_outputs[comb]
#                                for comb in torch.where(class_combination)[0]]
#             for attr in class_for_imply:
#                 attr = attr.replace("::", "_")
#                 rule += attr[:-1] + " v "
#             rule = rule[:-3]
#         rules.append(rule)
#
#     # Attribute --> Classes
#     main_class_outputs = [class_names[i] for i in main_classes]
#     for j_a, j in enumerate(attributes):
#         a = class_names[j]
#         a = a.replace("::", "_")
#         rule = a[:-1] + " -> "
#         attribute_combination = torch.tensor(combinations[:, j_a], dtype=torch.bool)
#         if torch.sum(attribute_combination.to(torch.int)) > 0:
#             output_for_imply = [main_class_outputs[comb]
#                                 for comb in torch.where(attribute_combination)[0]]
#             for o in output_for_imply:
#                 rule += o[:-1] + " v "
#             rule = rule[:-3]
#         rules.append(rule)
#
#     for rule in rules:
#         print(rule)
#
#     # OR on the main classes
#     output_or = ""
#     for m_c in main_class_outputs:
#         output_or += m_c[:-1] + " v "
#     output_or = output_or[:-3]
#     rules.append(output_or)
#
#     print(output_or)
#
#     # OR on the attributes
#     attr_or = ""
#     for a_c in attribute_outputs:
#         a_c = a_c.replace("::", "_")
#         attr_or += a_c[:-1] + " v "
#     attr_or = attr_or[:-3]
#     rules.append(attr_or)
#
#     print(attr_or)
#
#
# if __name__ == "__main__":
#     from data.Cub200 import CUBDataset
#
#     dataset = CUBDataset(os.path.join("../data", "cub200"),
#                          transforms.Compose(transforms.ToTensor()))
#     c_names = dataset.class_names
#
#     write_cub_rules(c_names, dataset.main_classes,
#                     dataset.attributes, dataset.class_attr_comb)
