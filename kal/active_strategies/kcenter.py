from typing import Tuple, List

import numpy as np
import torch

from kal.active_strategies import Strategy


class KCenterSampling(Strategy):

    def loss(self, preds, *args, x: torch.Tensor = None, labelled_idx: list = None, **kwargs) \
            -> torch.Tensor:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling loss method"
        assert labelled_idx is not None, "Labelled idx need to be passed " \
                                          "in KMeans Sampling loss method"
        k_center = x[labelled_idx]

        distances = np.linalg.norm(x[:, None, ...] - k_center[None, ...], axis=-1)
        distances = np.min(distances, axis=1)

        return torch.as_tensor(distances)

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  x: torch.Tensor = None, **kwargs) -> Tuple[List, torch.Tensor]:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling selection method"

        n_sample = preds.shape[0]
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        labelled_idx = np.asarray(labelled_idx)
        x = x.detach().numpy()

        dist_mat = np.matmul(x, x.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(n_sample, 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[avail_idx, :][:, labelled_idx]

        selected_idx = []
        for i in range(n_p):
            mat_min = mat.min(axis=1)
            i_center = mat_min.argmax()
            i_center_idx = avail_idx[i_center]
            selected_idx.append(i_center_idx)

            avail_idx = avail_idx[avail_idx != i_center_idx]
            labelled_idx = np.append(labelled_idx, i_center_idx)

            mat = np.delete(mat, i_center, 0)
            mat = np.append(mat, dist_mat[avail_idx, i_center][:, None], axis=1)

        loss = self.loss(preds, x=x, labelled_idx=labelled_idx)

        return selected_idx, loss

