from abc import ABC, abstractmethod
from typing import Union, Tuple, List

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from torch import Tensor
from torch.utils.data import TensorDataset
from torch_explain.logic import replace_names, test_explanation
from torch_explain.logic.nn.entropy import explain_class
from torch_explain.nn import EntropyLinear
import torch.nn.functional as F

from kal.network import MLP, train_loop


class XAI(ABC):
    @abstractmethod
    def explain(self, *args):
        raise NotImplementedError()

    def explain_cv(self, *args):
        raise NotImplementedError()


class XAI_ELEN(XAI):

    def __init__(self, hidden_size, epochs, lr, class_names, dev):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.class_names = class_names
        self.dev = dev

    def explain(self, x, labels, train_idx, labelled_idx) -> List:
        formulas = []
        input_size = x.shape[1]
        multi_dataset = TensorDataset(x[train_idx], labels[train_idx])
        expl_feats, expl_labels = x[train_idx] > 0.5, labels[train_idx]
        expl_model = ELEN(input_size=input_size, hidden_size=self.hidden_size,
                          n_classes=2, dropout=True).to(self.dev)
        train_loop(expl_model, multi_dataset, labelled_idx, self.epochs, lr=self.lr)
        expl_acc = expl_model.score(expl_feats, expl_labels)
        if expl_acc < 0.9:
            print(f"Low expl_accs: {expl_acc} Error in training the explainer")
        for i in range(2):
            formula = expl_model.explain(expl_feats, expl_labels,
                                         feat_names=self.class_names, target_class=i)
            formulas.append(formula)

        return formulas

    def explain_cv_multi_class(self, n_classes, labels, labelled_idx) -> List:
        from kal.network import evaluate
        formulas = []
        for i in range(n_classes):
            expl_feats = torch.cat([labels[:, :i], labels[:, i + 1:]], dim=1)
            expl_label = labels[:, i].unsqueeze(dim=1)
            expl_dataset = TensorDataset(expl_feats, expl_label)

            expl_model = ELEN(input_size=n_classes - 1, hidden_size=self.hidden_size,
                              n_classes=1, dropout=True).to(self.dev)
            train_loop(expl_model, expl_dataset, labelled_idx, self.epochs, lr=self.lr, device=expl_feats.device)
            expl_acc = evaluate(expl_model, expl_dataset, device=expl_feats.device)[0]
            if expl_acc < 0.9:
                print(f"Low expl_accs: {expl_acc} Error in training the explainer")
            formula = expl_model.explain(expl_feats, expl_label, self.class_names,
                                         target_class=0)
            formulas.append(formula)
        return formulas


class XAI_TREE(XAI):

    def __init__(self, *args, class_names, height=None, discretize_feats=False, **kwargs):
        self.class_names = class_names
        self.height = height
        self.discretize_feats = discretize_feats

    def explain(self, x, labels, labelled_idx) -> List:
        from kal.utils import tree_to_formula
        np.random.seed(0)

        formulas = []
        classes = range(labels.shape[1]) if len(labels.squeeze().shape) > 1 else [1]
        expl_feats, expl_labels = x[labelled_idx].cpu(), labels[labelled_idx].cpu() > 0.5
        if self.discretize_feats:
            expl_feats = expl_feats > 0.5
        expl_model = DecisionTreeClassifier(max_depth=self.height)
        expl_model.fit(expl_feats, expl_labels)
        expl_acc = f1_score(expl_labels,
                            expl_model.predict(expl_feats), average="macro", zero_division=1)
        if expl_acc < 0.9 and len(labelled_idx) > 10 and not self.discretize_feats:
            print(f"Low expl_accs: {expl_acc} Error in training the explainer")
        for i in classes:
            formula = tree_to_formula(expl_model, self.class_names, target_class=i,
                                      discretize_feats=self.discretize_feats)
            formulas.append(formula)
        return formulas

    def explain_cv(self, n_classes, labels, labelled_idx, main_classes) -> List:
        from kal.utils import tree_to_formula
        np.random.seed(0)

        expl_accs = []
        formulas = []
        expl_feats = labels[:, len(main_classes):].cpu().numpy()
        if self.discretize_feats:
            expl_feats = expl_feats > 0.5
        expl_label = labels[:, :len(main_classes)].argmax(dim=1).cpu().numpy()
        expl_names = self.class_names[len(main_classes):]
        expl_model = DecisionTreeClassifier(max_depth=self.height)
        expl_model.fit(expl_feats[labelled_idx], expl_label[labelled_idx])
        expl_accs += [f1_score(expl_label,
                               expl_model.predict(expl_feats), average="macro", zero_division=1)]
        for i in main_classes:
            formula = tree_to_formula(expl_model, expl_names, target_class=i)
            formulas.append(formula)

        expl_feats = labels[:, :len(main_classes)].cpu().numpy()
        if self.discretize_feats:
            expl_feats = expl_feats > 0.5
        expl_label = labels[:, len(main_classes):].cpu().numpy() > 0.5
        expl_names = self.class_names[:len(main_classes)]
        for i in range(n_classes - len(main_classes)):
            expl_model = DecisionTreeClassifier(max_depth=self.height)
            expl_model.fit(expl_feats[labelled_idx], expl_label[labelled_idx, i])
            expl_accs += [f1_score(expl_label[:, i],
                                   expl_model.predict(expl_feats), average="macro", zero_division=1)]
            formula = tree_to_formula(expl_model, expl_names,
                                      target_class=1, skip_negation=True)
            formulas.append(formula)
        if np.mean(np.asarray(expl_accs)) < 0.8:
            print(f"Low expl_accs: {np.mean(np.asarray(expl_accs))} "
                  f"Error in training the explainer")

        return formulas

    def explain_cv_multi_class(self, n_classes, labels, labelled_idx) -> List:
        from kal.utils import tree_to_formula
        np.random.seed(0)

        expl_accs = []
        formulas, formulas2 = [], []
        train_labels = labels[labelled_idx].cpu().numpy()
        for i in range(n_classes):
            non_i_classes = np.asarray([j for j in range(n_classes) if j != i])
            expl_label = train_labels[:, i] > 0.5
            expl_feats = train_labels[:, non_i_classes]
            if self.discretize_feats:
                expl_feats = expl_feats > 0.5
            expl_names = np.asarray(self.class_names)[non_i_classes]
            expl_model = DecisionTreeClassifier(max_depth=self.height)
            expl_model.fit(expl_feats, expl_label)
            expl_accs += [f1_score(expl_label, expl_model.predict(expl_feats), average="macro", zero_division=1)]
            formula = tree_to_formula(expl_model, expl_names, target_class=1)
            formulas += [formula]
            formulas2 += [f"{self.class_names[i]} <-> {formula}"]

        return formulas

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

    def explain(self, x, y, feat_names, target_class, return_acc=False):
        train_mask = torch.ones(x.shape[0], dtype=bool)
        expl_raw = explain_class(self, x, y, train_mask, train_mask,
                                 target_class=target_class, y_threshold=0.5, try_all=False)[0]
        expl = replace_names(expl_raw, feat_names)
        if return_acc:
            accuracy, _ = test_explanation(expl_raw, x.cpu(), y.cpu(), target_class)
            return expl, accuracy
        return expl


if __name__ == "__main__":
    x = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    # OR FORMULA
    output = torch.tensor([
        0.0,
        1.0,
        1.0,
        1.0,
    ])
    xai = XAI_TREE(class_names=["x1", "x2"])
    expl = xai.explain(x, output, [0, 1, 2, 3])
    assert expl[1] == '(x2 <= 0.5 & x1 > 0.5) | (x2 > 0.5)', f"Error in computing formula {expl}"

    # AND FORMULA
    output = torch.tensor([
        0.0,
        0.0,
        0.0,
        1.0,
    ])
    xai = XAI_TREE(class_names=["x1", "x2"])
    expl = xai.explain(x, output, [0, 1, 2, 3])
    assert expl[1] == '(x2 > 0.5 & x1 > 0.5)', f"Error in computing formula {expl}"

    # AND WITH NEG FORMULA
    output = torch.tensor([
        0.0,
        1.0,
        0.0,
        0.0,
    ])
    xai = XAI_TREE(class_names=["x1", "x2"])
    expl = xai.explain(x, output, [0, 1, 2, 3])
    assert expl[1] == '(x2 > 0.5 & x1 <= 0.5)', f"Error in computing formula {expl}"
