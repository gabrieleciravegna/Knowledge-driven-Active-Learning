from typing import Tuple, List, Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor

from .strategy import Strategy


class KMeansSampling(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.cluster_learner = None
        self.dis = None
        self.cluster_idx = None

    def loss(self, *args, x: torch.Tensor = None, n_cluster: int = None,
             return_clusters=False, **kwargs) -> Union[Tuple[Tensor, np.ndarray], Tensor]:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling loss method"
        assert n_cluster is not None, "Number of clusters needs to passed "\
                                      "in KMeans Sampling loss method"

        x = x.detach().numpy()
        if self.dis is None:
            cluster_learner = KMeans(n_clusters=n_cluster)
            cluster_learner.fit(x)

            cluster_idx = cluster_learner.predict(x).astype(int)
            self.cluster_idx = cluster_idx

            centers = cluster_learner.cluster_centers_[cluster_idx]
            dis = (x - centers) ** 2
            dis = dis.sum(axis=1)

            self.dis = torch.as_tensor(dis)

        if return_clusters:
            return self.dis, self.cluster_idx
        return self.dis

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  x: torch.Tensor = None, **kwargs) -> Tuple[List, torch.Tensor]:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling selection method"
        n_sample = preds.shape[0]

        # calculate the distances to each cluster. the number of cluster is equal to n_p
        dis, cluster_idx = self.loss(n_cluster=n_p, x=x.cpu(),
                                     return_clusters=True)
        cluster_idx[torch.as_tensor(labelled_idx)] = -1

        # select for each cluster the sample closest to it in the unlabelled pool
        selected_idx = []
        sample_idx = np.arange(n_sample)
        while len(selected_idx) != n_p:         # in case there are empty clusters
            for i in range(n_p):
                cluster_i_samples = cluster_idx == i
                if np.sum(cluster_i_samples) != 0:
                    sample_min_dist = dis[cluster_i_samples].argmin()
                    selected_idx.append(sample_idx[cluster_i_samples][sample_min_dist])
                    if len(selected_idx) == n_p:
                        break

        assert torch.as_tensor([idx not in labelled_idx for idx in selected_idx]).all(), \
            "Error: selected idx already labelled"

        assert len(selected_idx) == n_p, "Error in the selection of the idx"

        return selected_idx, dis
