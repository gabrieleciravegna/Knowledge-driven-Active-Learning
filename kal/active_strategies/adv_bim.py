from typing import Tuple, List

import torch
import torch.nn.functional as F

from kal.active_strategies.strategy import Strategy


class AdversarialBIMSampling(Strategy):

    def __init__(self, *args, eps=0.05, k_sample=1000, max_iter=100, main_classes=None, **kwargs):
        assert main_classes is not None, "Need to pass the list of main classes"

        super(AdversarialBIMSampling, self).__init__(*args, **kwargs)
        self.main_classes = main_classes
        self.eps = eps
        self.k_sample = k_sample
        self.max_iter = max_iter

    def loss(self, preds: torch.Tensor, *args, clf: torch.nn.Module = None,
             x: torch.Tensor = None, **kwargs) -> torch.Tensor:
        assert clf is not None, "Need to pass the classifier in the Adv DeepFool selection"
        assert x is not None, "Need to pass the Input data in the Adv DeepFool selection"
        assert len(preds.shape) > 1, "Adversarial Sampling requires multi-class prediction"

        dev = next(clf.parameters()).device
        dis = torch.zeros(x.shape[0])
        for j in range(x.shape[0]):
            x_j = x[j]
            nx = torch.unsqueeze(x_j, 0).to(dev)
            nx.requires_grad_()
            eta = torch.zeros(nx.shape).to(dev)

            _, out = clf(nx + eta, return_logits=True)
            out = out[:, self.main_classes]
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

                _, out = clf(nx + eta, return_logits=True)
                out = out[:, self.main_classes]
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
        rand_idx = torch.randperm(n_sample)[:self.k_sample]
        rand_x = x[rand_idx]

        adv_loss = self.loss(preds, *args, clf=clf, x=rand_x, **kwargs)

        labelled_rand_idx = [i for i, idx in enumerate(rand_idx)
                             if idx in labelled_idx]
        if len(labelled_rand_idx) > 0:
            adv_loss[torch.as_tensor(labelled_rand_idx)] = 1e30

        adv_idx = torch.argsort(adv_loss)
        adv_idx = rand_idx[adv_idx]
        adv_idx = adv_idx[:n_p].detach().cpu().numpy().tolist()

        assert torch.as_tensor([idx not in labelled_idx for idx in adv_idx]).all(), \
            "Error: selected idx already labelled"

        return adv_idx, adv_loss
