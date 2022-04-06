from typing import Tuple, List

import numpy as np
import torch
from torch import tensor

from kal.active_strategies.strategy import Strategy


class AdversarialDeepFoolSampling(Strategy):

    def __init__(self, *args, max_iter=50, k_sample=1000,
                 **kwargs):
        self.max_iter = max_iter
        self.k_sample = k_sample
        super(AdversarialDeepFoolSampling, self).__init__(*args, **kwargs)

    def loss(self, preds, *args, clf: torch.nn.Module = None,  x: torch.Tensor = None,
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
            n_class = out.shape[1]
            py = out.max(1)[1].item()
            ny = out.max(1)[1].item()

            i_iter = 0

            while py == ny and i_iter < self.max_iter:
                out[0, py].backward(retain_graph=True)
                grad_np = nx.grad.data.clone()
                value_l = np.inf
                ri = None

                for i in range(n_class):
                    if i == py:
                        continue

                    nx.grad.data.zero_()
                    out[0, i].backward(retain_graph=True)
                    grad_i = nx.grad.data.clone()

                    wi = grad_i - grad_np
                    fi = out[0, i] - out[0, py]
                    value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                    if value_i < value_l:
                        ri = value_i / np.linalg.norm(wi.numpy().flatten()) * wi

                eta += ri.clone() if ri is not None else 0.
                nx.grad.data.zero_()
                out = clf(nx + eta, logits=True)
                py = out.max(1)[1].item()
                i_iter += 1

            eta = eta.detach()
            dis[j] = (eta * eta).sum()
        return dis

    # def loss(self, preds, *args, clf: torch.nn.Module = None, x: torch.Tensor = None,
    #          **kwargs) -> torch.Tensor:
    #     assert clf is not None, "Need to pass the classifier in the Adv DeepFool selection"
    #     assert x is not None, "Need to pass the Input data in the Adv DeepFool selection"
    #     assert len(preds.shape) > 1, "Adversarial Sampling requires multi-class prediction"
    #     dis = torch.zeros(x.shape[0])
    #     for j in range(x.shape[0]):
    #         x_j = x[j]
    #         nx = torch.unsqueeze(x_j, 0)
    #         nx.requires_grad_()
    #         eta = torch.zeros(nx.shape).to(nx.device)
    #
    #         out = clf(nx + eta)
    #         n_class = out.shape[1]
    #         py = torch.argmax(out, dim=1)
    #         ny = py.clone()
    #
    #         i_iter = 0
    #
    #         while py == ny and i_iter < self.max_iter:
    #             out[0, py].backward(retain_graph=True)
    #             grad_np = nx.grad.data.clone()
    #             value_l = torch.inf
    #             ri = None
    #
    #             for i in range(n_class):
    #                 if i == py:
    #                     continue
    #
    #                 nx.grad.data.zero_()
    #                 out[0, i].backward(retain_graph=True)
    #                 grad_i = nx.grad.data.clone()
    #
    #                 wi = grad_i - grad_np
    #                 fi = out[0, i] - out[0, py]
    #                 value_i = torch.abs(fi) / torch.norm(wi)
    #
    #                 if value_i < value_l:
    #                     ri = value_i / torch.norm(wi) * wi
    #
    #             if ri is not None:
    #                 eta += ri.clone()
    #             else:
    #                 eta += 0.
    #
    #             nx.grad.data.zero_()
    #             out = clf(nx + eta)
    #             py = out.max(1)[1]
    #             i_iter += 1
    #
    #         eta = eta.detach()
    #         dis[j] = (eta * eta).sum()
    #     return dis
    #
    # def loss(self, preds, *args, clf: torch.nn.Module = None, x: torch.Tensor = None,
    #          **kwargs) -> torch.Tensor:
    #     assert clf is not None, "Need to pass the classifier in the Adv DeepFool selection"
    #     assert x is not None, "Need to pass the Input data in the Adv DeepFool selection"
    #     assert len(preds.shape) > 1, "Adversarial Sampling requires multi-class prediction"
    #
    #     nx = x
    #     dev = nx.device
    #     # nx = torch.unsqueeze(x_j, 0)
    #     nx.requires_grad_()
    #     eta = torch.zeros(nx.shape).to(dev)
    #
    #     out = clf(nx + eta)
    #     n_class = out.shape[1]
    #     py = out.max(1)[1]
    #     ny = out.max(1)[1]
    #
    #     i_iter = 0
    #
    #     while (py == ny).any() and i_iter < self.max_iter:
    #         subselect_out = out[range(out.shape[0]), py]
    #         subselect_out.backward(torch.ones_like(subselect_out), retain_graph=True)
    #         grad_np = nx.grad.data.clone()
    #         value_l = torch.inf
    #         ri = None
    #
    #         for i in range(n_class):
    #             # if i == py:
    #             #     continue
    #             mask = py != i
    #
    #             nx.grad.data.zero_()
    #             out[:, i].backward(mask, retain_graph=True)
    #             grad_i = nx.grad.data.clone()
    #
    #             wi = grad_i - grad_np
    #             fi = out[:, i] - out[range(out.shape[0]), py]
    #             value_i = torch.abs(fi) / torch.norm(wi)
    #
    #             if value_i < value_l:
    #                 ri = value_i / torch.norm(wi) * wi
    #
    #         # subselect only data that have not changed class yet
    #         eta = torch.where(py == ny, eta, tensor(0).to(dev))
    #         eta += ri.clone() if ri is not None else 0
    #         nx.grad.data.zero_()
    #         out = clf(nx + eta)
    #         py = out.max(1)[1]
    #         i_iter += 1
    #
    #     eta = eta.detach()
    #     dis = (eta * eta).sum()
    #     return dis

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
