from typing import Tuple, Union

import numpy as np
import torch

from . import KnowledgeLoss


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
