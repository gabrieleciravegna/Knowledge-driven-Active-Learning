import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt

from kal.knowledge import KnowledgeLoss
from kal.network import MLP


def visualize_data_predictions(network: MLP, data: TensorDataset,
                               k_loss: KnowledgeLoss, idx: list = None):

    input_data = data.tensors[0]
    labels = data.tensors[1]
    x_0, x_1 = input_data[:, 0].numpy(), input_data[:, 1].numpy()
    with torch.no_grad():
        p_t = network(input_data).squeeze()
    p = p_t.numpy()
    k_l = k_loss(p_t, x=input_data)
    s_loss = torch.nn.BCELoss(reduction="none")(p_t, labels)
    if idx is None:
        idx = [*range(input_data.shape[0])]
    idx = np.asarray(idx)
    sns.scatterplot(x=x_0[idx], y=x_1[idx], hue=labels[idx].numpy())
    plt.title("Selected data points")
    plt.xlim([0, 1]), plt.ylim([0, 1]), plt.show()
    sns.scatterplot(x=x_0, y=x_1, hue=p)
    plt.title("Predictions of the network")
    plt.xlim([0, 1]), plt.ylim([0, 1]), plt.show()
    sns.scatterplot(x=x_0, y=x_1, hue=k_l)
    plt.title("Constraint Loss")
    plt.xlim([0, 1]), plt.ylim([0, 1]), plt.show()
    sns.scatterplot(x=x_0, y=x_1, hue=s_loss)
    plt.title("Supervision Loss")
    plt.xlim([0, 1]), plt.ylim([0, 1]), plt.show()
    return
