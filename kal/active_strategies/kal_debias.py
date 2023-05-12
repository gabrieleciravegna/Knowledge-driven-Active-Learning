from typing import List, Tuple, Union

import torch
import seaborn as sns
from matplotlib import pyplot as plt
from kal.active_strategies.kal_xai import KALXAISampling


class KALDEBIASSampling(KALXAISampling):
    def selection(self, *args, **kwargs):
        labels = kwargs['labels']
        bias = kwargs['bias']
        kwargs['formulas'] = [bias] if not isinstance(bias, list) else bias
        # if kwargs['biased_model']:
        args = args[1:]
        active_idx, active_loss = super().selection(labels, *args, **kwargs)
        # else:
        #     active_idx, active_loss = super().selection(*args, **kwargs)

        active_sample_bias = 1 - active_loss[active_idx].mean()
        if active_sample_bias > 0.5:
            x = kwargs['x']
            sns.scatterplot(x=x[active_idx, 0], y=x[active_idx, 1], hue=labels[active_idx])
            plt.show()
            raise ValueError(f"Error in the debias selection strategy. "
                             f"Bias in the selected data: {active_sample_bias}")
        return active_idx, active_loss


class KALDEBIASUncSampling(KALDEBIASSampling):
    def loss(self, *args, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        kwargs['uncertainty'] = True
        return super(KALDEBIASUncSampling, self).loss(*args, **kwargs)


class KALDEBIASDiversitySampling(KALDEBIASSampling):

    def selection(self, *args, **kwargs) -> Tuple[List, torch.Tensor]:
        if "diversity" in kwargs:
            kwargs.pop("diversity")
        return super().selection(*args, diversity=True, **kwargs)


class KALDEBIASDiversityUncSampling(KALDEBIASDiversitySampling):
    def loss(self, *args, **kwargs) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        kwargs['uncertainty'] = True
        return super(KALDEBIASDiversityUncSampling, self).loss(*args, **kwargs)
