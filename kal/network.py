from typing import Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch_explain.logic import replace_names
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, resnet18
from tqdm import trange
from torch_explain.nn import EntropyLinear
from torch_explain.logic.nn.entropy import explain_class

from kal.losses import CombinedLoss
from kal.metrics import Metric, F1

num_workers = 0


class ResNet18(torch.nn.Module):
    def __init__(self, n_classes, transfer_learning=False, pretrained=True):
        super().__init__()
        self.model = resnet18(pretrained=pretrained, progress=True)

        if transfer_learning:
            for param in self.model.parameters():
                param.requires_grad = False
        feat_dim = self.model.fc.weight.shape[1]
        self.model.fc = torch.nn.Linear(feat_dim, n_classes)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x, return_logits=False):
        logits = self.model(x)
        output = self.activation(logits)

        if return_logits:
            return output, logits
        return output


class ResNet10(torch.nn.Module):
    def __init__(self, n_classes, transfer_learning=False):
        super().__init__()
        self.model = ResNet(BasicBlock, [1, 1, 1, 1])
        if transfer_learning:
            for param in self.model.parameters():
                param.requires_grad = False
        feat_dim = self.model.fc.weight.shape[1]
        self.model.fc = torch.nn.Linear(feat_dim, n_classes)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x, return_logits=False):
        logits = self.model(x)
        output = self.activation(logits)

        if return_logits:
            return output, logits
        return output


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
        # if n_classes > 1:
        #     self.activation = torch.nn.Softmax()
        # else:
        self.activation = torch.nn.Sigmoid()
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
        output = self.activation(logits)

        if return_logits:
            return output, logits
        return output


class ELEN(MLP):
    def __init__(self, *args, **kwargs):
        super(ELEN, self).__init__(*args, **kwargs)
        self.fc1 = EntropyLinear(self.input_size, self.hidden_size,
                                 self.n_classes, temperature=1)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, input_x: torch.Tensor, return_logits=False) -> Union[Tuple[Tensor, Tensor], Tensor]:
        hidden = self.fc1(input_x)
        relu = self.relu(hidden)

        if self.dropout:
            dropout = F.dropout(relu, p=self.dropout_rate)
        else:
            dropout = relu
        logits = self.fc2(dropout).squeeze(-1)
        output = self.activation(logits)

        if return_logits:
            return output, logits
        return output

    def explain(self, x, y, train_idx, feat_names, target_class):
        expl = explain_class(self, x, y, train_idx, train_idx,
                             target_class=target_class, y_threshold=0.5)[0]
        expl = replace_names(expl, feat_names)
        return expl


def train_loop(network: torch.nn.Module, data: TensorDataset, train_idx: List,
               epochs: int = 100, batch_size=None, lr=1e-2,
               loss=torch.nn.BCEWithLogitsLoss(reduction="none"),
               optimizer=torch.optim.AdamW, visualize_loss: bool = False,
               device=torch.device("cpu"), verbose=False) \
        -> List[Tensor]:

    network.to(device)
    train_idx = np.asarray(train_idx, dtype=int)
    train_data = Subset(data, train_idx)

    if batch_size is None:
        data_loader = [[tensor[train_data.indices]
                        for tensor in train_data.dataset.tensors]]
    else:
        data_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True,
                                 num_workers=num_workers, shuffle=True)
    optimizer = optimizer(network.parameters(), lr=lr)

    l_train = []
    network.train()
    pbar = trange(epochs) if verbose else range(epochs)
    for _ in pbar:
        for input_data, labels in data_loader:
            input_data, labels = input_data.to(device), labels.to(device)
            optimizer.zero_grad()
            output, logits = network(input_data, return_logits=True)
            if isinstance(loss, CombinedLoss):
                s_l = loss(logits, target=labels, x=input_data)
            else:
                s_l = loss(logits, labels)
            # s_l[s_l > 1] /= 10  # may allow to avoid numerical problems
            s_l = s_l.mean()
            s_l.backward()
            optimizer.step()
            l_train.append(s_l)

    network.eval()
    if verbose:
        pbar.close()

    l_train = torch.stack(l_train).detach().cpu().tolist()
    if visualize_loss:
        sns.lineplot(data=l_train)
        plt.ylabel("Loss"), plt.xlabel("Epochs")
        plt.title("Training loss variations in function of the epochs")
        plt.show()

    assert l_train[-1] < 10, f"Error in fitting the data. " \
                             f"High training loss {l_train[-1]:.2f}." \
                             f"Try reducing the learning rate (current lr: {lr:.4f}"
    if l_train[-1] > 1.:
        print(f"Error in fitting the data. High training loss {l_train[-1]:.2f}."
              f"Try reducing the learning rate (current lr: {lr:.4f}")

    return l_train


def predict(network, data: TensorDataset, batch_size=None, loss=None,
            device=torch.device("cpu")) \
        -> Union[Tuple[Tensor, List], Tensor]:
    network.eval()
    network.to(device)

    if batch_size is None:
        data_loader = [[tensor for tensor in data.tensors]]
    else:
        data_loader = DataLoader(data, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    preds = []
    losses = []
    with torch.no_grad():
        for input_data, labels in data_loader:
            input_data, labels = input_data.to(device), labels.to(device)
            p_t, logits = network(input_data, return_logits=True)
            preds.append(p_t.squeeze())
            if loss is not None:
                l_val = loss(logits, target=labels)
                if len(l_val.shape) > 1:
                    l_val = l_val.sum(dim=1)
                losses += l_val.cpu().numpy().tolist()
    preds = torch.cat(preds)

    if loss is not None:
        return preds, losses
    return preds


def predict_dropout(network, data: TensorDataset, batch_size=None,
                    device=torch.device("cpu"), n_splits=5) -> Tensor:
    network.train()
    network.to(device)

    if batch_size is None:
        data_loader = [[tensor for tensor in data.tensors]]
    else:
        data_loader = DataLoader(data, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)

    with torch.no_grad():
        preds_drop = []
        for _ in range(n_splits):
            preds = []
            for input_data, labels in data_loader:
                input_data, labels = input_data.to(device), labels.to(device)
                p_t = network(input_data)
                preds.append(p_t.squeeze())
            preds = torch.cat(preds)
            preds_drop.append(preds)
        preds_drop = torch.stack(preds_drop).mean(dim=0)

    return preds_drop


def predict_dropout_splits(network, data: TensorDataset, batch_size=None,
                           n_splits=10, device=torch.device("cpu")) -> Tensor:
    network.train()
    network.to(device)

    if batch_size is None:
        data_loader = [[tensor for tensor in data.tensors]]
    else:
        data_loader = DataLoader(data, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)

    split_preds = []
    for i in range(n_splits):
        preds = []
        with torch.no_grad():
            for input_data, labels in data_loader:
                input_data, labels = input_data.to(device), labels.to(device)
                p_t = network(input_data)
                preds.append(p_t.squeeze())
        preds = torch.cat(preds)
        split_preds.append(preds)
    split_preds = torch.stack(split_preds)
    return split_preds


def evaluate(network: MLP, data: TensorDataset,
             batch_size=None, loss=torch.nn.BCELoss(reduction="none"),
             metric: Metric = None, device=torch.device("cpu"),
             return_preds=False) \
        -> Union[Tuple[float, Tensor, Tensor], Tuple[float, Tensor]]:

    if metric is None:
        metric = F1()

    if isinstance(data, TensorDataset):
        labels = data.tensors[1]
    elif isinstance(data, Subset):
        labels = torch.as_tensor(data.dataset.targets[data.indices])
    else:
        labels = torch.as_tensor(data.targets)

    preds, l_test = predict(network, data, batch_size, loss, device)

    accuracy = metric(preds, labels)

    if return_preds:
        return accuracy, l_test, preds

    return accuracy, l_test
