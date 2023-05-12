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
from sklearn.tree import _tree, DecisionTreeClassifier
from sympy import to_dnf, simplify_logic, true

epsilon = 1e-10


def visualize_data_predictions(x_t: torch.Tensor, itr: int, act_strategy: str,
                               dataframe: pd.DataFrame, png_file: str = None,
                               dimensions=None, seed=0, dataset="xor",
                               active_loss=False, bias=False):
    from kal.active_strategies import NAME_MAPPINGS
    if dimensions is None:
        dimensions = [0, 1]
    dataframe = dataframe[dataframe["Seed"] == seed]
    df_strategy = dataframe[dataframe["Strategy"] == act_strategy].reset_index()
    df_iteration = df_strategy[df_strategy['Iteration'] == itr]

    a_idx = df_iteration["Active Idx"].item()
    u_idx = df_iteration["Used Idx"].item()
    # train_idx = df_iteration["Train Idx"].item()
    n_points = len(u_idx)
    assert n_points == len(np.unique(u_idx)), f"Error in selecting points {u_idx}"

    x_0, x_1 = x_t.cpu().numpy()[:, dimensions[0]], \
               x_t.cpu().numpy()[:, dimensions[1]]
    preds = df_iteration["Predictions"].item()

    # multi_class = False
    if dataset == "xor":
        if len(preds.shape) > 1:
            preds = preds[:, 1]
        new_idx = [1 if idx in a_idx else 0 for idx in u_idx]
        if active_loss:
            a_loss = df_iteration["Active Loss"].item()
            sns.scatterplot(x=x_0, y=x_1, hue=a_loss, legend=True)
        elif bias:
            bias_measure = df_iteration["Bias Loss"].item()
            bias_measure[bias_measure < 1e-10] = 0
            sns.scatterplot(x=x_0, y=x_1, hue=bias_measure, legend=True)
        else:
            sns.scatterplot(x=x_0, y=x_1, hue=preds, legend=True)
            sns.scatterplot(x=x_0[np.asarray(u_idx)], y=x_1[np.asarray(u_idx)],
                            hue=new_idx, size=new_idx, sizes=[70, 100], legend=False)
        plt.axhline(0.5, 0, 1, c="k")
        plt.axvline(0.5, 0, 1, c="k")
        plt.xlabel("$x_0$")
        plt.ylabel("$x_1$")
    else:
        preds = np.argmax(preds, axis=1)
        new_idx = [1 if idx in a_idx else 0 for idx in u_idx]
        sns.set_palette(sns.color_palette()[:3])
        sns.scatterplot(x=x_0, y=x_1, hue=preds, legend=True)
        sns.scatterplot(x=x_0[np.asarray(u_idx)], y=x_1[np.asarray(u_idx)], legend=False,
                        hue=preds[u_idx], size=new_idx, sizes=[150, 200], style=new_idx)
        # new_idx = [2 if idx in a_idx else 1 if idx in u_idx else 0
        #            for idx in range(preds.shape[0])]
        # sns.scatterplot(x=x_0, y=x_1, hue=preds,
        #                 style=new_idx, markers=['o', 'X', 'D', ])
        plt.xlabel("$Petal Length$")
        plt.ylabel("$Petal Width$")

    # plt.title(f"Points selected by {act_strategy}, iter {itr}, "
    #           f"acc {acc:.2f}, n_points{n_points}")
    plt.title(f"{NAME_MAPPINGS[act_strategy]} - it {itr} - seed {seed} ")
    # f"- {n_points} p - {acc:.2f} %", fontsize=28)
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
        f_name = re.sub('[^a-zA-Z0-9_&|~ \n\.]', '', f_name)
        mapping.append((f_name, f_abbr))

    explanation = re.sub('[^a-zA-Z0-9_&<>=|~ \n\.]', '', explanation)
    for k, v in mapping:
        # explanation = explanation.replace(k, v)
        explanation = re.sub(r"\b%s\b" % k, v, explanation)

    return explanation


def tree_to_formula(tree: DecisionTreeClassifier, concept_names: List[str], target_class: int,
                    skip_negation=False, discretize_feats=False, simplify=True) -> str:
    """
    Translate a decision tree into a set of decision rules.

    :param tree: sklearn decision tree
    :param concept_names: concept names
    :param target_class: target class
    :param skip_negation:
    :return: decision rule
    """
    tree_ = tree.tree_
    feature_name = [
        concept_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    pathto = dict()

    global k
    global explanation
    explanation = ''
    k = 0

    def recurse(node, depth, parent):
        global k
        global explanation
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # assert threshold == 0.5, f"Error in threshold {threshold}"
            s = f'{name} <= {threshold}' if not discretize_feats else f'~{name}'
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent] + ' & ' + s
            recurse(tree_.children_left[node], depth + 1, node)

            s = f'{name} > {threshold}' if not discretize_feats else f'{name}'
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent] + ' & ' + s
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            k = k + 1
            try:
                if len(tree_.value[node].squeeze().shape) == 1:
                    node_class = tree_.value[node].squeeze().argmax()
                else:
                    node_class = tree_.value[node][:,1].argmax()
                if node_class == target_class:
                    explanation += f'({pathto[parent]}) | '
            except:
                print("error in extracting formula")
                pass

    recurse(0, 1, 0)
    explanation = explanation[:-3]
    if skip_negation:
        new_expl = ""
        for or_term in explanation.split(" | "):
            new_expl += "("
            for and_term in or_term.split(" & "):
                if "~" not in and_term:
                    new_expl += and_term.replace(")", "", ).replace("(", "") + " & "
            if new_expl == "(":
                new_expl = or_term + " | "
            else:
                new_expl = new_expl[:-3] + ") | "
        explanation = new_expl[:-3]
    if simplify and explanation != "" and discretize_feats:
        explanation = simplify_logic(explanation)
        explanation = str(explanation)

    return explanation


def check_bias_in_exp(formula, bias):
    # dnf_formula = str(to_dnf(formula, simplify=True, force=True))
    # if len(dnf_formula.split(" | ")) != len(bias.split(" | ")) == 1:
    #     return False
    # if simplify_logic(f"(({formula}) & ({bias})) | "
    #                   f"((~{bias}) & ~({formula}))"):
    if formula == "True":
        return True
    if simplify_logic(formula) == simplify_logic(bias):
        return True

    return False


def inv_steep_sigmoid(x: torch.Tensor, k=100, b=0.5) -> torch.Tensor:
    output: torch.Tensor = 1 / (1 + torch.exp(k * (x - b)))
    return output


def steep_sigmoid(x: torch.Tensor, k=100, b=0.5) -> torch.Tensor:
    output: torch.Tensor = 1 / (1 + torch.exp(-k * (x - b)))
    return output


def double_implication_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    impl_1 = a * (1 - b)
    impl_2 = b * (1 - a)
    return impl_1 + impl_2