from typing import Tuple, Union

import numpy as np
import torch

from kal.knowledge import KnowledgeLoss


class DogvsPersonLoss(KnowledgeLoss):
    def __init__(self, names=None, scale=None, mu=1, **kwargs):
        super().__init__(names)
        self.scale = scale
        self.mu = mu

    def __call__(self, output, return_arg_max=False, targets=False, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        Dog = output[:, 0]
        Dog_ear = output[:, 1]
        Dog_head = output[:, 2]
        Dog_leg = output[:, 3]
        Dog_muzzle = output[:, 4]
        Dog_neck = output[:, 5]
        Dog_nose = output[:, 6]
        Dog_paw = output[:, 7]
        Dog_tail = output[:, 8]
        Dog_torso = output[:, 9]
        Person = output[:, 10]
        Person_arm = output[:, 11]
        Person_foot = output[:, 12]
        Person_hair = output[:, 13]
        Person_hand = output[:, 14]
        Person_head = output[:, 15]
        Person_leg = output[:, 16]
        Person_neck = output[:, 17]
        Person_nose = output[:, 18]
        Person_torso = output[:, 19]

        loss_fol_product_tnorm = []

        # if a:
        loss_fol_product_tnorm.extend([
            # A: OBJECT-PART --> [OBJECTS] RULES
            (Dog_ear * (1 - Dog)),
            (Dog_head * (1 - Dog)),
            (Dog_leg * (1 - Dog)),
            (Dog_muzzle * (1 - Dog)),
            (Dog_neck * (1 - Dog)),
            (Dog_nose * (1 - Dog)),
            (Dog_paw * (1 - Dog)),
            (Dog_tail * (1 - Dog)),
            (Dog_torso * (1 - Dog)),

            (Person_arm * (1 - Person)),
            (Person_foot * (1 - Person)),
            (Person_hair * (1 - Person)),
            (Person_hand * (1 - Person)),
            (Person_head * (1 - Person)),
            (Person_leg * (1 - Person)),
            (Person_neck * (1 - Person)),
            (Person_nose * (1 - Person)),
            (Person_torso * (1 - Person)),
        ])

        # if b:
        loss_fol_product_tnorm.extend([
            # B: OBJECT --> [OBJECT-PARTS] RULES

            (Dog *
             (1 - Dog_ear) *
             (1 - Dog_head) *
             (1 - Dog_leg) *
             (1 - Dog_muzzle) *
             (1 - Dog_neck) *
             (1 - Dog_nose) *
             (1 - Dog_paw) *
             (1 - Dog_tail) *
             (1 - Dog_torso)
             ),

            (Person *
             (1 - Person_arm) *
             (1 - Person_foot) *
             (1 - Person_hair) *
             (1 - Person_hand) *
             (1 - Person_head) *
             (1 - Person_leg) *
             (1 - Person_neck) *
             (1 - Person_nose) *
             (1 - Person_torso)
             ),

        ])

        # if c:
        # C: OR ON THE OBJECTS
        loss_fol_product_tnorm.extend([
            ((1 - Dog) * (1 - Person))
         ])

        # if d:
        # D: OR ON THE PARTS
        loss_fol_product_tnorm.extend([(
                (1 - Dog_ear) *
                (1 - Dog_head) *
                (1 - Dog_leg) *
                (1 - Dog_muzzle) *
                (1 - Dog_neck) *
                (1 - Dog_nose) *
                (1 - Dog_paw) *
                (1 - Dog_tail) *
                (1 - Dog_torso) *
                (1 - Person_arm) *
                (1 - Person_foot) *
                (1 - Person_hair) *
                (1 - Person_hand) *
                (1 - Person_head) *
                (1 - Person_leg) *
                (1 - Person_neck) *
                (1 - Person_nose) *
                (1 - Person_torso)
        )])

        losses = torch.stack(loss_fol_product_tnorm, dim=1)

        if self.scale:
            if self.scale == "a" or self.scale == "both":
                # scale the first group of rules for the number of predictions made
                # (they may become noisy)
                num_preds = (output > 0.5).sum(dim=1)
                scaling = np.ones(output.shape[0]) / (num_preds + 1)  # to avoid numerical problem
                scaled_losses = losses[:44] * scaling
                losses[:44] = scaled_losses
            if self.scale == "c" or self.scale == "both":
                # scale by a factor 10 the penultimate rule (which is the most important)
                losses[-2] = losses[-2] * self.mu

        # losses = torch.sum(losses, dim=1)

        loss_sum = torch.squeeze(torch.sum(losses, dim=1))

        threshold = 0.5 if targets else 10.
        self.check_loss(loss_sum, losses.T, loss_sum, threshold)

        # print("Output", output)
        # print("Losses", losses)
        # print("Loss_sum", loss_sum)

        if return_arg_max:
            arg_max = torch.argmax(losses, dim=0)
            return loss_sum, arg_max
        return loss_sum

    @staticmethod
    def get_rules():

        rules = [
            # A: OBJECT-PART --> [OBJECTS] RULES
            "Dog_ear -> Dog",
            "Dog_head -> Dog",
            "Dog_leg -> Dog",
            "Dog_muzzle -> Dog",
            "Dog_neck -> Dog",
            "Dog_nose -> Dog",
            "Dog_paw -> Dog",
            "Dog_tail -> Dog",
            "Dog_torso -> Dog",

            "Person_arm -> Person",
            "Person_foot -> Person",
            "Person_hair -> Person",
            "Person_hand -> Person",
            "Person_head -> Person",
            "Person_leg -> Person",
            "Person_neck -> Person",
            "Person_nose -> Person",
            "Person_torso -> Person",

            # B: OBJECT --> [OBJECT-PARTS] RULES

            "Dog -> Dog_ear | Dog_head | Dog_leg | Dog_muzzle | "
            "Dog_neck | Dog_nose | Dog_paw | Dog_tail | Dog_torso",

            "Person -> Person_arm  | Person_foot  | Person_hair  | "
            "Person_hand  | Person_head  | Person_leg  | Person_neck  | "
            "Person_nose  | Person_torso",

            # C: OR ON THE OBJECTS
            "Dog | Person",

            # D: OR ON THE PARTS
            "Dog_ear | Dog_head | Dog_leg | Dog_muzzle | Dog_neck | "
            "Dog_nose | Dog_paw | Dog_tail | Dog_torso | Person_arm | "
            "Person_foot | Person_hair | Person_hand | Person_head | "
            "Person_leg | Person_neck | Person_nose | Person_torso",
        ]

        return rules


if __name__ == "__main__":
    from kal.utils import to_latex

    list_rules = DogvsPersonLoss.get_rules()

    to_latex(list_rules, "dog_rules.txt", truncate=False)
