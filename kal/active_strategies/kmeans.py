from typing import Tuple, List, Any, Union

import numpy as np
import torch
from torch import Tensor

from .strategy import Strategy
from sklearn.cluster import KMeans


class KMeansSampling(Strategy):

    def loss(self, preds, *args, x: torch.Tensor = None, n_cluster: int = None,
             return_clusters=False, **kwargs) -> Union[Tuple[Tensor, np.ndarray], Tensor]:
        assert x is not None, "Input data/Embeddings need to be passed " \
                              "in KMeans Sampling loss method"
        assert n_cluster is not None, "Number of clusters needs to passed "\
                                      "in KMeans Sampling loss method"

        x = x.detach().numpy()
        cluster_learner = KMeans(n_clusters=n_cluster)
        cluster_learner.fit(x)

        cluster_idx = cluster_learner.predict(x).astype(int)
        centers = cluster_learner.cluster_centers_[cluster_idx]
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
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        avail_preds = preds[avail_idx]
        avail_x = x[avail_idx]

        # calculate the distances to each cluster. the number of cluster is equal to n_p
        dis, cluster_idx = self.loss(avail_preds, n_cluster=n_p, x=avail_x,
                                     return_clusters=True)

        # select for each cluster the sample closest to it in the unlabelled pool
        selected_idx = [avail_idx[cluster_idx == i][dis[cluster_idx == i].argmin()]
                        for i in range(n_p)]

        return selected_idx, dis
