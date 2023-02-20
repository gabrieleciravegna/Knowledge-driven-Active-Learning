import random
from typing import Tuple, Union

import torch

from . import KnowledgeLoss


class AnimalLoss(KnowledgeLoss):
    def __init__(self, names, mu=1, uncertainty=False, percentage=None):
        super().__init__(names)
        self.mu = mu
        self.uncertainty = uncertainty
        self.percentage = percentage

    def __call__(self, output, targets=False, return_argmax=False, return_losses=False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # MAIN CLASSES
        ALBATROSS = output[:, 0]
        CHEETAH = output[:, 1]
        GIRAFFE = output[:, 2]
        OSTRICH = output[:, 3]
        PENGUIN = output[:, 4]
        TIGER = output[:, 5]
        ZEBRA = output[:, 6]

        # ATTRIBUTE CLASSES
        BIRD = output[:, 7]
        BLACK = output[:, 8]
        BLACKSTRIPES = output[:, 9]
        BLACKWHITE = output[:, 10]
        CARNIVORE = output[:, 11]
        CLAWS = output[:, 12]
        CUD = output[:, 13]
        DARKSPOTS = output[:, 14]
        EVENTOED = output[:, 15]
        FEATHER = output[:, 16]
        FLY = output[:, 17]
        FORWARDEYES = output[:, 18]
        GOODFLIER = output[:, 19]
        HAIR = output[:, 20]
        HOOFS = output[:, 21]
        LAYEGGS = output[:, 22]
        LONGLEGS = output[:, 23]
        LONGNECK = output[:, 24]
        MAMMAL = output[:, 25]
        MEAT = output[:, 26]
        MILK = output[:, 27]
        POINTEDTEETH = output[:, 28]
        SWIM = output[:, 29]
        TAWNY = output[:, 30]
        UNGULATE = output[:, 31]
        WHITE = output[:, 32]

        # here we converted each FOL rule using the product T-Norm (no-residual)
        losses = [
            # 0) HAIR => MAMMAL
            (HAIR * (1. - MAMMAL)),
            # 1) MILK => MAMMAL
            (MILK * (1. - MAMMAL)),
            # 2) FEATHER => BIRD
            (FEATHER * (1. - BIRD)),
            # 3) FLY ^ LAYEGGS => BIRD
            ((FLY * LAYEGGS) * (1. - BIRD)),
            # 4) MAMMAL ^ MEAT => CARNIVORE
            ((MAMMAL * MEAT) * (1. - CARNIVORE)),
            # 5) MAMMAL ^ POINTEDTEETH ^ CLAWS ^ FORWARDEYES => CARNIVORE
            ((MAMMAL * POINTEDTEETH * CLAWS * FORWARDEYES) * (1. - CARNIVORE)),
            # 6) MAMMAL ^ HOOFS => UNGULATE
            ((MAMMAL * HOOFS) * (1. - UNGULATE)),
            # 7) MAMMAL ^ CUD => UNGULATE
            ((MAMMAL * CUD) * (1. - UNGULATE)),
            # 8) MAMMAL ^ CUD => EVENTOED
            ((MAMMAL * CUD) * (1. - EVENTOED)),
            # 9)CARNIVORE ^ TAWNY ^ DARKSPOTS => CHEETAH
            ((CARNIVORE * TAWNY * DARKSPOTS) * (1. - CHEETAH)),
            # 10)CARNIVORE ^ TAWNY ^ BLACKWHITE => TIGER
            ((CARNIVORE * TAWNY * BLACKWHITE) * (1. - TIGER)),
            # 11) UNGULATE ^ LONGLEGS ^ LONGNECK ^ TAWNY ^ DARKSPOTS => GIRAFFE
            ((UNGULATE * LONGLEGS * LONGNECK * TAWNY * DARKSPOTS) * (1. - GIRAFFE)),
            # 12) BLACKSTRIPES ^ UNGULATE ^ WHITE => ZEBRA
            ((BLACKSTRIPES * UNGULATE * WHITE) * (1. - ZEBRA)),
            # 13) BIRD ^ !FLY ^ LONGLEGS ^ LONGNECK ^ BLACK => OSTRICH
            ((BIRD * (1. - FLY) * LONGLEGS * LONGNECK * BLACK) * (1. - OSTRICH)),
            # 14) BIRD ^ !FLY ^ LONGLEGS ^ SWIM ^ BLACKWHITE => PENGUIN
            ((BIRD * (1. - FLY) * SWIM * BLACKWHITE) * (1. - PENGUIN)),
            # 15) BIRD ^ GOODFLIER => ALBATROSS
            ((BIRD * GOODFLIER) * (1. - ALBATROSS)),

            # 16) XOR ON THE MAIN CLASSES
            self.mu * ((1 - (ALBATROSS * (1 - CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) * (1 - TIGER) * (
                    1 - ZEBRA))) *
                       (1 - ((1 - ALBATROSS) * CHEETAH * (1 - GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) * (1 - TIGER) * (
                               1 - ZEBRA))) *
                       (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * GIRAFFE * (1 - OSTRICH) * (1 - PENGUIN) * (1 - TIGER) * (
                               1 - ZEBRA))) *
                       (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * OSTRICH * (1 - PENGUIN) * (1 - TIGER) * (
                               1 - ZEBRA))) *
                       (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * PENGUIN * (1 - TIGER) * (
                               1 - ZEBRA))) *
                       (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) * TIGER * (
                               1 - ZEBRA))) *
                       (1 - ((1 - ALBATROSS) * (1 - CHEETAH) * (1 - GIRAFFE) * (1 - OSTRICH) * (1 - PENGUIN) *
                             (1 - TIGER) * ZEBRA))),

            # 17) OR ON THE ATTRIBUTE CLASSES
            self.mu * ((1 - BIRD) * (1 - BLACK) * (1 - BLACKSTRIPES) * (1 - BLACKWHITE) * (1 - CARNIVORE) *
                       (1 - CLAWS) * (1 - CUD) * (1 - DARKSPOTS) * (1 - EVENTOED) * (1 - FEATHER) * (1 - FLY) *
                       (1 - FORWARDEYES) * (1 - GOODFLIER) * (1 - HAIR) * (1 - HOOFS) * (1 - LAYEGGS) *
                       (1 - LONGLEGS) * (1 - LONGNECK) * (1 - MAMMAL) * (1 - MEAT) * (1 - MILK) * (1 - POINTEDTEETH) *
                       (1 - SWIM) * (1 - TAWNY) * (1 - UNGULATE) * (1 - WHITE))
        ]
        if self.percentage is not None:
            n_rules = round(len(losses) * self.percentage // 100)
            losses = random.sample(losses, k=n_rules)

        if self.uncertainty:
            unc_loss = 0
            for i in range(output.shape[1]):
                unc_loss += output[:, i] * (1 - output[:, i])
            losses.append(unc_loss)

        losses = torch.stack(losses, dim=1)
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
