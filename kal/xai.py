from abc import ABC, abstractmethod
from typing import Union, Tuple, List

import torch
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

    def explain_cv(self, n_classes, labels, labelled_idx) -> List:
        expl = [ELEN(input_size=n_classes - 1, hidden_size=self.hidden_size,
                     n_classes=1, dropout=True).to(self.dev) for _ in range(n_classes)]
        formulas = []
        for i in range(n_classes):
            expl_feats = torch.cat([labels[:, :i], labels[:, i + 1:]], dim=1)
            expl_label = labels[:, i].unsqueeze(dim=1)
            expl_dataset = TensorDataset(expl_feats, expl_label)
            train_loop(expl[i], expl_dataset, labelled_idx, self.epochs, lr=self.lr)
            formula = expl[i].explain(expl_feats, expl_label, self.class_names,
                                      target_class=0)
            formulas.append(formula)


class XAI_TREE(XAI):

    def __init__(self, *args, class_names, **kwargs):
        self.class_names = class_names
        pass

    def explain(self, x, labels, labelled_idx) -> List:
        from kal.utils import tree_to_formula

        formulas = []
        n_classes = labels.shape[1] if len(labels.shape) > 1 else 1
        expl_feats, expl_labels = x > 0.5, labels
        expl_model = DecisionTreeClassifier()
        expl_model = expl_model.fit(expl_feats[labelled_idx], expl_labels[labelled_idx])
        expl_acc = expl_model.score(expl_feats, expl_labels)
        if expl_acc < 0.9:
            print(f"Low expl_accs: {expl_acc} Error in training the explainer")
        for i in range(n_classes):
            formula = tree_to_formula(expl_model, self.class_names, target_class=i)
            formulas.append(formula)
        return formulas

    def explain_cv(self, n_classes, labels, labelled_idx, main_classes) -> List:
        from kal.utils import tree_to_formula

        expl_accs = []
        formulas = []
        expl_model = DecisionTreeClassifier()
        expl_feats = labels[:, len(main_classes):].cpu().numpy()
        expl_label = labels[:, :len(main_classes)].argmax(dim=1).cpu().numpy()
        expl_names = self.class_names[len(main_classes):]
        expl_model = expl_model.fit(expl_feats[labelled_idx], expl_label[labelled_idx])
        expl_accs.append(expl_model.score(expl_feats, expl_label))
        for i in main_classes:
            formula = tree_to_formula(expl_model, expl_names, target_class=i)
            formulas.append(formula)

        expl_feats = labels[:, :len(main_classes)].cpu().numpy()
        expl_label = labels[:, len(main_classes):].cpu().numpy()
        expl_names = self.class_names[:len(main_classes)]
        for i in range(n_classes - len(main_classes)):
            expl_model = DecisionTreeClassifier()
            expl_model = expl_model.fit(expl_feats[labelled_idx], expl_label[labelled_idx, i])
            expl_accs.append(expl_model.score(expl_feats, expl_label[:, i]))
            formula = tree_to_formula(expl_model, expl_names, target_class=1, skip_negation=True)
            formulas.append(formula)
        assert expl_accs[0] > 0.9, "Error in training the explainer"

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
