from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Subset
import seaborn as sns
import matplotlib.pyplot as plt

from kal.metrics import MultiLabelAccuracy, Metric


class MLP(torch.nn.Module):
    def __init__(self, n_classes, input_size, hidden_size, dropout=False, multi_class=False):
        super(MLP, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.n_classes)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = dropout

    def forward(self, input_x: torch.Tensor, logits=False):
        hidden = self.fc1(input_x)
        relu = self.relu(hidden)
        if self.dropout:
            dropout = F.dropout(relu, p=0.1)
        else:
            dropout = relu
        output = self.fc2(dropout)
        if logits:
            return output
        output = self.sigmoid(output)
        return output


def train_loop(network: MLP, data: TensorDataset, train_idx: list,
               epochs: int = 100, batch_size=None, lr=1e-2,
               loss=torch.nn.BCELoss(reduction="none"),
               optimizer=torch.optim.AdamW, visualize_loss: bool = False) \
        -> list[Tensor]:

    train_idx = np.asarray(train_idx)
    train_data = Subset(data, train_idx)

    if batch_size is None:
        batch_size = len(data)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = optimizer(network.parameters(), lr=lr)

    l_train = []
    network.train()
    for j in range(epochs):
        for input_data, labels in data_loader:
            optimizer.zero_grad()
            p_t = network(input_data).squeeze()
            s_l = loss(p_t, labels)
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
    return l_train


def predict(network, data: TensorDataset, batch_size=None):
    network.eval()

    if batch_size is None:
        batch_size = len(data)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for input_data, y in data_loader:
            p_t = network(input_data).squeeze()
            preds.append(p_t)
    preds = torch.cat(preds)

    return preds


def predict_dropout(network, data: TensorDataset, batch_size=None):
    network.train()

    if batch_size is None:
        batch_size = len(data)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for input_data, y in data_loader:
            p_t = network(input_data).squeeze()
            preds.append(p_t)
    preds = torch.cat(preds)

    return preds


def predict_dropout_splits(network, data: TensorDataset, batch_size=None, n_splits=10):
    if batch_size is None:
        batch_size = len(data)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    split_preds = []
    for i in range(n_splits):
        preds = []
        with torch.no_grad():
            for input_data, y in data_loader:
                p_t = network(input_data).squeeze()
                preds.append(p_t)
        preds = torch.cat(preds)
        split_preds.append(preds)
    split_preds = torch.stack(split_preds)
    return split_preds


def evaluate(network: MLP, data: TensorDataset,
             batch_size=None, loss=torch.nn.BCELoss(reduction="none"),
             metric: Metric = None) -> Tuple[float, Tensor, Tensor]:

    n_classes = torch.unique(data.tensors[1])
    if metric is None:
        metric = MultiLabelAccuracy(n_classes)

    labels = data.tensors[1]

    preds = predict(network, data, batch_size)

    l_test = loss(preds, labels)
    if len(l_test.shape) > 1:
        l_test = l_test.sum(dim=1)

    accuracy = metric(preds, labels)

    return accuracy, preds, l_test
