from typing import Tuple, List

import numpy as np
import torch

from kal.active_strategies import Strategy


def delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i != ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)


class KCenterSampling(Strategy):

    def __init__(self, *args, k_sample=1e4, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_sample=1e4

    def loss(self, preds, *args, x: torch.Tensor = None, labelled_idx: list = None, **kwargs) \
            -> torch.Tensor:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling loss method"
        assert labelled_idx is not None, "Labelled idx need to be passed " \
                                          "in KMeans Sampling loss method"
        k_center = x[labelled_idx]

        if x.shape[0] + x.shape[1] < 1e6:
            return torch.zeros(x.shape[0])

        distances = torch.norm(x[:, None, ...] - k_center[None, ...], dim=-1)
        distances = torch.min(distances, dim=1)[0]

        return distances

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  x: torch.Tensor = None, **kwargs) -> Tuple[List, torch.Tensor]:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling selection method"

        n_sample = preds.shape[0]
        k_sample = np.min([self.k_sample, n_sample])

        rand_idx = torch.randperm(n_sample).numpy()[:k_sample]
        for idx in labelled_idx:
            if idx not in rand_idx:
                rand_idx = np.append(rand_idx, idx)
        rand_labelled_idx = np.asarray([np.argwhere(rand_idx == idx)[0][0] for idx in labelled_idx])
        k_sample = rand_idx.shape[0]

        rand_x = x[rand_idx]

        labelled_idx_bool = np.zeros(k_sample, dtype=bool)
        labelled_idx_bool[rand_labelled_idx] = True

        dist_mat = torch.matmul(rand_x.cpu(), rand_x.T.cpu())
        sq = torch.diagonal(dist_mat).reshape(k_sample, 1).clone()
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.T
        dist_mat = torch.sqrt(dist_mat)

        mat = dist_mat[~labelled_idx_bool, :][:, labelled_idx_bool]

        for _ in range(n_p):
            mat_min = mat.min(dim=1)[0]
            i_center = mat_min.argmax()
            i_center_idx = np.arange(k_sample)[~labelled_idx_bool][i_center]
            labelled_idx_bool[i_center_idx] = True
            mat = delete(mat, i_center, 0)
            mat = torch.cat((mat, dist_mat[~labelled_idx_bool, i_center_idx][:, None]), dim=1)

        selected_idx = [idx for idx in np.where(labelled_idx_bool)[0] if idx not in rand_labelled_idx]
        selected_idx = rand_idx[selected_idx].tolist()

        assert torch.as_tensor([idx not in labelled_idx for idx in selected_idx]).all(), \
            "Error: selected idx already labelled"

        loss = self.loss(preds, x=x, labelled_idx=selected_idx)

        return selected_idx, loss

