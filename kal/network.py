from typing import Tuple, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Subset
import seaborn as sns
import matplotlib.pyplot as plt

from kal.metrics import MultiLabelAccuracy, Metric


class MLP(torch.nn.Module):
    def __init__(self, n_classes, input_size, hidden_size, dropout=False,
                 dropout_rate=0.01):
        super(MLP, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.n_classes)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = dropout
        self.dropout_rate = dropout_rate

    def forward(self, input_x: torch.Tensor, return_logits=False) \
            -> Union[Tuple[Tensor, Tensor], Tensor]:
        hidden = self.fc1(input_x)
        relu = self.relu(hidden)

        if self.dropout:
            dropout = F.dropout(relu, p=self.dropout_rate)
        else:
            dropout = relu
        logits = self.fc2(dropout)
        output = self.sigmoid(logits)

        if return_logits:
            return output, logits
        return output


def train_loop(network: MLP, data: TensorDataset, train_idx: List,
               epochs: int = 100, batch_size=None, lr=1e-2,
               loss=torch.nn.BCEWithLogitsLoss(reduction="none"),
               optimizer=torch.optim.AdamW, visualize_loss: bool = False) \
        -> List[Tensor]:

    train_idx = np.asarray(train_idx)
    train_data = Subset(data, train_idx)

    if batch_size is None:
        data_loader = [[tensor[train_data.indices] for tensor in train_data.dataset.tensors]]
    else:
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = optimizer(network.parameters(), lr=lr)

    l_train = []
    network.train()
    for j in range(epochs):
        for input_data, labels in data_loader:
            optimizer.zero_grad()
            _, logits = network(input_data, return_logits=True)
            s_l = loss(logits.squeeze(), labels)
            # s_l[s_l > 1] /= 10  # may allow to avoid numerical problems
            s_l = s_l.mean()
            s_l.backward()
            optimizer.step()
            l_train.append(s_l.item())
    network.eval()

    if visualize_loss:
        sns.lineplot(data=l_train)
        plt.ylabel("Loss"), plt.xlabel("Epochs")
        plt.title("Training loss variations in function of the epochs")
        plt.show()

    assert l_train[-1] < 10, f"Error in fitting the data. High training loss {l_train[-1]:.2f}." \
                             f"Try reducing the learning rate (current lr: {lr:.4f}"
    return l_train


def predict(network, data: TensorDataset, batch_size=None, loss=None) \
        -> Union[Tuple[Tensor, List], Tensor]:
    network.train()
    if batch_size is None:
        data_loader = [[tensor for tensor in data.tensors]]
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    preds = []
    losses = []
    with torch.no_grad():
        for input_data, y in data_loader:
            p_t, logits = network(input_data, return_logits=True)
            preds.append(p_t.squeeze())
            if loss is not None:
                l_val = loss(logits.squeeze(), y)
                if len(l_val.shape) > 1:
                    l_val = l_val.sum(dim=1)
                losses += l_val.cpu().numpy().tolist()
    preds = torch.cat(preds)

    if loss is not None:
        return preds, losses
    return preds


def predict_dropout(network, data: TensorDataset, batch_size=None) -> Tensor:
    network.train()
    if batch_size is None:
        data_loader = [[tensor for tensor in data.tensors]]
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for input_data, y in data_loader:
            p_t = network(input_data)
            preds.append(p_t.squeeze())
    preds = torch.cat(preds)

    return preds


def predict_dropout_splits(network, data: TensorDataset, batch_size=None,
                           n_splits=10) -> Tensor:
    if batch_size is None:
        data_loader = [[tensor for tensor in data.tensors]]
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    split_preds = []
    for i in range(n_splits):
        preds = []
        with torch.no_grad():
            for input_data, y in data_loader:
                p_t = network(input_data)
                preds.append(p_t.squeeze())
        preds = torch.cat(preds)
        split_preds.append(preds)
    split_preds = torch.stack(split_preds)
    return split_preds


def evaluate(network: MLP, data: TensorDataset,
             batch_size=None, loss=torch.nn.BCELoss(reduction="none"),
             metric: Metric = None) -> Tuple[float, Tensor, List]:

    n_classes = torch.unique(data.tensors[1])
    if metric is None:
        metric = MultiLabelAccuracy(n_classes)

    labels = data.tensors[1]

    preds, l_test = predict(network, data, batch_size, loss)

    accuracy = metric(preds, labels)

    return accuracy, preds, l_test
