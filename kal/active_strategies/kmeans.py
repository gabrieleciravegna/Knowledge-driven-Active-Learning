from typing import Tuple, List, Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor

from .strategy import Strategy


class KMeansSampling(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_learner = None

    def loss(self, *args, x: torch.Tensor = None, n_cluster: int = None,
             return_clusters=False, **kwargs) -> Union[Tuple[Tensor, np.ndarray], Tensor]:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling loss method"
        assert n_cluster is not None, "Number of clusters needs to passed "\
                                      "in KMeans Sampling loss method"

        x = x.detach().numpy()
        if self.cluster_learner is None:
            cluster_learner = KMeans(n_clusters=n_cluster)
            cluster_learner.fit(x)
            self.cluster_learner = cluster_learner

        cluster_idx = self.cluster_learner.predict(x).astype(int)
        centers = self.cluster_learner.cluster_centers_[cluster_idx]

        dis = (x - centers) ** 2
        dis = dis.sum(axis=1)

        if return_clusters:
            return torch.as_tensor(dis), cluster_idx
        return torch.as_tensor(dis)

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int, *args,
                  x: torch.Tensor = None, **kwargs) -> Tuple[List, torch.Tensor]:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling selection method"
        n_sample = preds.shape[0]

        # calculate the distances to each cluster. the number of cluster is equal to n_p
        dis, cluster_idx = self.loss(n_cluster=n_p, x=x,
                                     return_clusters=True)
        dis[torch.as_tensor(labelled_idx)] = 1e30

        # select for each cluster the sample closest to it in the unlabelled pool
        selected_idx = [np.arange(n_sample)[cluster_idx == i][dis[cluster_idx == i].argmin()]
                        for i in range(n_p)]

        assert torch.as_tensor([idx not in labelled_idx for idx in selected_idx]).all(), \
            "Error: selected idx already labelled"

        return selected_idx, dis
