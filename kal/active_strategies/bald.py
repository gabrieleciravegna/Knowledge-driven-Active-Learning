from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Subset, Dataset

from kal.active_strategies.strategy import Strategy
from kal.network import predict_dropout_splits


class BALDSampling(Strategy):
    def __init__(self, n_splits=5, *args, **kwargs):
        self.n_splits = n_splits
        super().__init__(*args, **kwargs)

    def loss(self, preds, *args, 
             dataset=None, clf=None, **kwargs) -> torch.Tensor:

        assert dataset is not None, "BALDSampling requires the dataset to compute the entropy over splits"
        assert clf is not None,  "BALDSampling requires to classifier to computer the predictions over splits"

        split_preds = predict_dropout_splits(clf, dataset, n_splits=self.n_splits)
        pb = split_preds.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-split_preds*torch.log(split_preds)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1

        return uncertainties

    def selection(self, preds: torch.Tensor, labelled_idx: list, n_p: int,
                  *args, dataset: Dataset = None, clf=None, **kwargs) -> Tuple[List, torch.Tensor]:
        assert dataset is not None, "BALDSampling requires the dataset to compute the entropy over splits"
        assert clf is not None,  "BALDSampling requires to classifier to computer the predictions over splits"

        n_sample = preds.shape[0]
        avail_idx = np.asarray(list(set(np.arange(n_sample)) - set(labelled_idx)))
        avail_data = Subset(dataset, avail_idx)
        
        e_loss = self.loss(preds, clf=clf, dataset=avail_data)

        e_idx = torch.argsort(e_loss, descending=True)
        e_idx = e_idx[:-len(labelled_idx)].detach().cpu().numpy().tolist()

        return e_idx[:n_p], e_loss
