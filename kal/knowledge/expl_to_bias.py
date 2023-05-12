import torch

from kal.knowledge import KnowledgeLoss


class Expl_2_Bias:
    def __init__(self, k_loss:KnowledgeLoss):
        self.k_loss = k_loss

    def __call__(self, output, target_class, *args, **kwargs):
        valid_output = output[torch.where(output[:, target_class])[0]]
        c_loss = self.k_loss(valid_output, *args, **kwargs)
        return 1 - c_loss
