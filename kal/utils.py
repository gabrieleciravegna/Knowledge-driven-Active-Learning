from typing import List

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import random

# from torch.utils.data.dataset import TensorDataset
# from kal.knowledge import KnowledgeLoss
# from kal.network import MLP
from kal.active_strategies import NAME_MAPPINGS


def visualize_data_predictions(x_t: torch.Tensor, itr: int, act_strategy: str,
                               dataframe: pd.DataFrame, png_file: str = None,
                               dimensions=None, seed=0, dataset="xor"):
    if dimensions is None:
        dimensions = [0, 1]
    dataframe = dataframe[dataframe["Seed"] == seed]
    df_strategy = dataframe[dataframe["Strategy"] == act_strategy].reset_index()
    df_iteration = df_strategy[df_strategy['Iteration'] == itr]

    a_idx = df_iteration["Active Idx"].item()
    u_idx = df_iteration["Used Idx"].item()
    acc = df_iteration["Accuracy"].item()
    train_idx = df_iteration["Train Idx"].item()
    n_points = len(u_idx)

    x_0, x_1 = x_t.cpu().numpy()[train_idx, dimensions[0]], \
               x_t.cpu().numpy()[train_idx, dimensions[1]]
    preds = df_iteration["Predictions"].item()

    # multi_class = False
    if dataset == "xor":
        if len(preds.shape) > 1:
            preds = preds[:, 0]
        new_idx = [1 if idx in a_idx else 0 for idx in u_idx]
        sns.scatterplot(x=x_0, y=x_1, hue=preds, legend=False)
        sns.scatterplot(x=x_0[np.asarray(u_idx)], y=x_1[np.asarray(u_idx)],
                        hue=new_idx, legend=False)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
    else:
        preds = np.argmax(preds, axis=1)
        new_idx = [2 if idx in a_idx else 1 if idx in u_idx else 0
                   for idx in range(preds.shape[0])]
        sns.scatterplot(x=x_0, y=x_1, hue=preds, style=new_idx, markers=['o', 'X', 'D',])
        plt.xlabel("$Petal Length$")
        plt.ylabel("$petal Width$")

    plt.axhline(0.5, 0, 1, c="k")
    plt.axvline(0.5, 0, 1, c="k")
    # plt.title(f"Points selected by {act_strategy}, iter {itr}, "
    #           f"acc {acc:.2f}, n_points{n_points}")
    plt.title(f"{NAME_MAPPINGS[act_strategy]}", fontsize=36)
    plt.xticks([0.0, 0.5, 1.0])
    plt.yticks([0.0, 0.5, 1.0])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if png_file is not None:
        plt.savefig(png_file)
    plt.show()


def visualize_active_vs_sup_loss(i, active_strategy, dataframe, png_file: str = None,
                                 lin_regression=False):
    dataframe = dataframe[dataframe["seed"] == 0]
    df_strategy = dataframe[dataframe["strategy"] == active_strategy].reset_index()
    df_iteration = df_strategy[df_strategy['iteration'] == i]
    df_prev_iteration = df_strategy[df_strategy['iteration'] == i - 1]
    if i == 0:
        c_loss = df_iteration["constraint_loss"].item()
        s_loss = df_iteration["supervision_loss"].item()
    else:
        c_loss = df_prev_iteration["constraint_loss"].item()
        s_loss = df_prev_iteration["supervision_loss"].item()
    if i != len(df_strategy["active_idx"]):
        a_idx = df_iteration["active_idx"].item()
        u_idx = df_iteration["used_idx"].item()
    else:
        a_idx = []
        u_idx = df_prev_iteration["used_idx"].item()
    new_idx = [1 if idx in a_idx else 0 for idx in np.arange(c_loss.shape[0])]

    sns.scatterplot(x=c_loss, y=s_loss, hue=new_idx,
                    palette=['gray', 'darkorange'], legend=False)
    # sns.scatterplot(x=c_loss[np.asarray(u_idx)], y=s_loss[np.asarray(u_idx)],
    #                 style=new_idx, legend=False)
    if lin_regression:
        m, b = np.polyfit(c_loss, s_loss, 1)
        x = np.arange(np.min(c_loss), np.max(c_loss), 0.01)
        plt.plot(x, m * x + b)
    plt.title(f"Selected data points for {active_strategy} training, iter {i}")
    plt.yscale('symlog')
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if png_file is not None:
        plt.savefig(png_file)
    plt.show()

    return


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

# def visualize_data_predictions(network: MLP, data: TensorDataset,
#                                k_loss: KnowledgeLoss, idx: list = None):
#
#     input_data = data.tensors[0]
#     labels = data.tensors[1]
#     x_0, x_1 = input_data[:, 0].numpy(), input_data[:, 1].numpy()
#     with torch.no_grad():
#         p_t = network(input_data).squeeze()
#     p = p_t.numpy()
#     k_l = k_loss(p_t, x=input_data)
#     s_loss = torch.nn.BCELoss(reduction="none")(p_t, labels)
#     if idx is None:
#         idx = [*range(input_data.shape[0])]
#     idx = np.asarray(idx)
#     sns.scatterplot(x=x_0[idx], y=x_1[idx], hue=labels[idx].numpy())
#     plt.title("Selected data points")
#     plt.xlim([0, 1]), plt.ylim([0, 1]), plt.show()
#     sns.scatterplot(x=x_0, y=x_1, hue=p)
#     plt.title("Predictions of the network")
#     plt.xlim([0, 1]), plt.ylim([0, 1]), plt.show()
#     sns.scatterplot(x=x_0, y=x_1, hue=k_l)
#     plt.title("Constraint Loss")
#     plt.xlim([0, 1]), plt.ylim([0, 1]), plt.show()
#     sns.scatterplot(x=x_0, y=x_1, hue=s_loss)
#     plt.title("Supervision Loss")
#     plt.xlim([0, 1]), plt.ylim([0, 1]), plt.show()
#     return


def to_latex(rules: list, latex_file: str, truncate=True, terms=6):
    rules = [rule.replace("&", "$\land$") for rule in rules]
    rules = [rule.replace("|", "$\lor$") for rule in rules]
    rules = [rule.replace("->", "$\Rightarrow$") for rule in rules]
    rules = [rule.replace("_", "\_") for rule in rules]

    i = 0
    with open(latex_file, "w") as f:
        for rule in rules:
            if rule == "":
                # f.write("\\midrule \n")
                i = 0
            else:
                i += 1
                if i >= terms and truncate:
                    if i == terms:
                        f.write(" & $\ldots$ $\ldots$ \\\\ \n")
                    continue
                f.write("$\\forall x$  &  ")
                for j, term in enumerate(rule.split(" ")):
                    if j == terms and truncate:
                        f.write(" $\ldots$ ")
                        break
                    f.write(f" {term}")
                f.write("\\\\ \n")


def replace_expl_names(explanation: str, concept_names: List[str]) -> str:
    import re
    """
    Replace names of concepts in a formula.
    :param explanation: formula
    :param concept_names: new concept names
    :return: Formula with renamed concepts
    """
    feature_abbreviations = [f'feature{i:010}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        mapping.append((f_name, f_abbr))

    for k, v in mapping:
        # explanation = explanation.replace(k, v)
        explanation = re.sub(r"\b%s\b" % k, v, explanation)

    return explanation