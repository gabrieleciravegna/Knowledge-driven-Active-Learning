from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from kal.active_strategies.strategy import Strategy


class AdversarialBIMSampling(Strategy):

    def __init__(self, *args, eps=0.05, k_sample=1000, max_iter=100, **kwargs):
        super(AdversarialBIMSampling, self).__init__(*args, **kwargs)
        self.eps = eps
        self.k_sample = k_sample
        self.max_iter = max_iter

    def loss(self, preds, *args, clf: torch.nn.Module = None, x: torch.Tensor = None,
             **kwargs) -> torch.Tensor:
        assert clf is not None, "Need to pass the classifier in the Adv DeepFool selection"
        assert x is not None, "Need to pass the Input data in the Adv DeepFool selection"
        assert len(preds.shape) > 1, "Adversarial Sampling requires multi-class prediction"

        dis = torch.zeros(x.shape[0])
        for j in range(x.shape[0]):
            x_j = x[j]
            nx = torch.unsqueeze(x_j, 0)
            nx.requires_grad_()
            eta = torch.zeros(nx.shape)

            out = clf(nx + eta, logits=True)
            py = out.max(1)[1]
            ny = out.max(1)[1]

            i_iter = 0
            while py.item() == ny.item() and i_iter < self.max_iter:
                i_iter += 1

                loss = F.cross_entropy(out, ny)
                loss.backward()
                if (torch.sign(nx.grad.data) == 0.).all():
                    eta = 1e2 * torch.ones_like(nx)
                    break

                eta += self.eps * torch.sign(nx.grad.data)
                nx.grad.data.zero_()

                out = clf(nx + eta, logits=True)
                py = out.max(1)[1]

            if i_iter == self.max_iter:
                eta = 1e2 * torch.ones_like(nx)

            dis[j] = (eta * eta).sum()
        return dis

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int,
                  *args, x: torch.Tensor = None, clf: torch.nn.Module = None,
                  **kwargs) -> Tuple[List, torch.Tensor]:
        assert clf is not None, "Need to pass the classifier in the Adv DeepFool selection"
        assert x is not None, "Need to pass the Input data in the Adv DeepFool selection"

        n_sample = preds.shape[0]
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        avail_preds = preds[avail_idx]
        rand_idx = torch.randperm(avail_idx.shape[0])[:self.k_sample]
        rand_x = x[rand_idx]

        adv_loss = self.loss(avail_preds, *args, clf=clf, x=rand_x, **kwargs)

        adv_idx = torch.argsort(adv_loss)
        adv_idx = rand_idx[adv_idx]
        adv_idx = adv_idx[:n_p].detach().cpu().numpy().tolist()

        return adv_idx, adv_loss
