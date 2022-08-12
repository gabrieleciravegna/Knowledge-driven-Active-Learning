import os
from typing import Tuple, Union

import numpy as np
import torch

from kal.knowledge import KnowledgeLoss


class CUB200Loss(KnowledgeLoss):
    def __init__(self, main_classes: list, attributes: list, combinations: list,
                 names=None, uncertainty=False, percentage=None):

        super().__init__(names)
        self.uncertainty = uncertainty
        self.main_classes = main_classes
        self.attributes = attributes
        self.combinations = combinations
        self.mu = 1.
        self.percentage = percentage

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

        if self.percentage is not None:
            n_rules = losses.shape[1] * self.percentage // 100
            losses = losses[:, :n_rules]

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

    def get_rules(self, class_names: list):
        rules = []
        combinations = np.asarray(self.combinations)
        class_names = [class_name.replace("::", "_") for class_name in class_names]
        class_names = np.asarray(class_names)
        attributes = np.asarray(self.attributes)
        main_classes = np.asarray(self.main_classes)
        for i in main_classes:
            class_combination = torch.tensor(combinations[i, :], dtype=torch.bool)
            main_class_name = class_names[i]
            if torch.sum(class_combination.to(torch.int)) > 0:
                attributes_implied = attributes[class_combination]
                attributes_names = class_names[attributes_implied]
                rule = main_class_name + " -> "
                for attributes_name in attributes_names:
                    rule += attributes_name + " & "
                rule = rule[:-3]
                rules.append(rule)
        rules.append("")

        # Attribute --> Classes
        for j_a, j in enumerate(attributes):
            attribute_name = class_names[j]
            attribute_combination = torch.tensor(combinations[:, j_a], dtype=torch.bool)
            if torch.sum(attribute_combination.to(torch.int)) > 0:
                class_implied = main_classes[attribute_combination]
                class_names_implied = class_names[class_implied]
                rule = attribute_name + " -> "
                for class_name in class_names_implied:
                    rule += class_name + " & "
                rule = rule[:-3]
                rules.append(rule)

        rules.append("")

        # OR on the main classes
        or_rule_main = ""
        for main_class_name in class_names[main_classes]:
            or_rule_main += main_class_name + " | "
        or_rule_main = or_rule_main[:-3]
        rules.append(or_rule_main)
        rules.append("")

        # OR on the attributes
        or_rule_attr = ""
        for attr_class_name in class_names[attributes]:
            or_rule_attr += attr_class_name + " | "
        or_rule_attr = or_rule_attr[:-3]
        rules.append(or_rule_attr)

        return rules


if __name__ == "__main__":
    from data.Cub200 import CUBDataset
    from torchvision.transforms import transforms
    from kal.utils import to_latex

    root_folder = os.path.join("..", "..", "data", "CUB200")
    dataset = CUBDataset(root_folder, transforms.Compose([transforms.ToTensor()]))

    cub_loss = CUB200Loss(main_classes=dataset.main_classes,
                          attributes=dataset.attributes,
                          combinations=dataset.class_attr_comb)

    list_rules = cub_loss.get_rules(dataset.class_names)

    to_latex(list_rules, "cub_rules.txt")

