from typing import Union, List, Tuple

import numpy as np
import torch

from .strategy import Strategy


class MarginSampling(Strategy):

    def loss(self, preds: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(preds.shape) == 1:
            preds = preds.unsqueeze(dim=1)
            preds = torch.hstack((preds, 1 - preds))

        preds_sorted, _ = preds.sort(descending=True)
        uncertainties = 1 - (preds_sorted[:, 0] - preds_sorted[:, 1])
        return uncertainties

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  m_loss: torch.Tensor = None, **kwargs) \
            -> Tuple[List, torch.Tensor]:

        n_sample = preds.shape[0]
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        avail_preds = preds[avail_idx]

        if m_loss is None:
            m_loss = self.loss(avail_preds, *args, **kwargs)

        m_idx = torch.argsort(m_loss, descending=True)
        m_idx = m_idx[:n_p].detach().cpu().numpy().tolist()

        return m_idx, m_loss


class MarginDropoutSampling(MarginSampling):

    def loss(self, _, *args, preds_dropout=None, **kwargs) -> torch.Tensor:

        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        return super().loss(preds_dropout, *args, **kwargs)

    def selection(self, preds, *args, preds_dropout=None, **kwargs) \
            -> Tuple[List, torch.Tensor]:

        assert preds_dropout is not None, "Need to pass predictions made with dropout to calculate this metric"

        m_loss = self.loss(preds, *args, preds_dropout=preds_dropout, **kwargs)

        return super().selection(preds_dropout, *args, m_loss=m_loss, **kwargs)
