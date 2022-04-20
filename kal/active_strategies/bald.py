from typing import Tuple, List

import torch
from torch.utils.data import TensorDataset

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
                  *args, dataset: TensorDataset = None, clf=None, **kwargs) -> Tuple[List, torch.Tensor]:
        assert dataset is not None, "BALDSampling requires the dataset to compute the entropy over splits"
        assert clf is not None,  "BALDSampling requires to classifier to computer the predictions over splits"

        b_loss = self.loss(preds, clf=clf, dataset=dataset)

        b_loss[torch.as_tensor(labelled_idx)] = -1
        
        b_idx = torch.argsort(b_loss, descending=True)
        b_idx = b_idx[:n_p].detach().cpu().numpy().tolist()

        assert torch.as_tensor([idx not in labelled_idx for idx in b_idx]).all(), \
            "Error: selected idx already labelled"
        
        return b_idx, b_loss
