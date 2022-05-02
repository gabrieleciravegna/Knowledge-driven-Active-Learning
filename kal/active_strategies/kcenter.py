from typing import Tuple, List

import numpy as np
import torch

from kal.active_strategies import Strategy


def delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i != ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)


class KCenterSampling(Strategy):

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
        labelled_idx_bool = np.zeros(n_sample, dtype=bool)
        labelled_idx_bool[labelled_idx] = True
        # x = x.detach().cpu().numpy()

        dist_mat = torch.matmul(x, x.T)
        sq = torch.diagonal(dist_mat).reshape(n_sample, 1).clone()
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.T
        dist_mat = torch.sqrt(dist_mat)

        mat = dist_mat[~labelled_idx_bool, :][:, labelled_idx_bool]

        for _ in range(n_p):
            mat_min = mat.min(axis=1)[0]
            i_center = mat_min.argmax()
            i_center_idx = np.arange(n_sample)[~labelled_idx_bool][i_center]
            labelled_idx_bool[i_center_idx] = True
            mat = delete(mat, i_center, 0)
            mat = torch.cat((mat, dist_mat[~labelled_idx_bool, i_center_idx][:, None]), dim=1)

        selected_idx = [idx for idx in np.where(labelled_idx_bool)[0] if idx not in labelled_idx]

        assert torch.as_tensor([idx not in labelled_idx for idx in selected_idx]).all(), \
            "Error: selected idx already labelled"

        loss = self.loss(preds, x=x, labelled_idx=labelled_idx_bool)

        return selected_idx, loss

        #
        # sq = np.array(dist_mat.diagonal()).reshape(n_sample, 1)
        # dist_mat *= -2
        # dist_mat += sq
        # dist_mat += sq.transpose()
        # dist_mat = np.sqrt(dist_mat)
        #
        # mat = dist_mat[avail_idx, :][:, labelled_idx]
        #
        # selected_idx = []
        # for i in range(n_p):
        #     mat_min = mat.min(axis=1)
        #     i_center = mat_min.argmax()
        #     i_center_idx = avail_idx[i_center]
        #     selected_idx.append(i_center_idx)
        #
        #     avail_idx = avail_idx[avail_idx != i_center_idx]
        #     labelled_idx = np.append(labelled_idx, i_center_idx)
        #
        #     mat = np.delete(mat, i_center, 0)
        #     mat = np.append(mat, dist_mat[avail_idx, i_center][:, None], axis=1)

