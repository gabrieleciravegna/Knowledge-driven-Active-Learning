from typing import Tuple, Union, Callable

import numpy as np
import torch


class KnowledgeLoss:

    def __init__(self, names=None):
        if names is not None:
            if not isinstance(names, np.ndarray):
                names = np.asarray(names)
        self.names = names

    def __call__(self, preds: torch.Tensor, return_argmax=False, return_losses=False, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError()

    def check_loss(self, output: torch.Tensor, losses: torch.Tensor,
                   loss_sum: torch.Tensor, threshold: float):

        if (loss_sum > threshold).any():
            indexes = torch.nonzero(loss_sum > threshold)
            for index in indexes:
                index = index.squeeze()
                strange_prediction = output[index]
                print(f"Very high constraint loss {loss_sum[index]} for index {index}")
                class_preds = strange_prediction > 0.5
                if self.names is not None:
                    name_strange_pred = self.names[class_preds.cpu().numpy()]
                    print(f"Class predicted: {name_strange_pred}")
                else:
                    print(f"Class idx predicted: {torch.where(class_preds)[0]}")
                violated_rules = torch.where(losses[:, index] > 0.5)
                print(f"Most violated rules {violated_rules}")

        assert loss_sum.min() >= 0.0, f"Error in calculating constraints, " \
                                      f"got negative loss {loss_sum.min()}"

