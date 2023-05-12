
# %% md

# Knowledge-Driven Active Learning - Experiment on the XOR problem

# %% md
import os

from kal.knowledge.expl_to_loss import Expl_2_Loss
from kal.xai import XAI_TREE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datetime
import random
import time
from statistics import mean
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from kal.knowledge import IrisLoss


from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils.data import TensorDataset

from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, ENTROPY_D, ENTROPY, ADV_DEEPFOOL, ADV_BIM, BALD, \
    DROPOUTS, KAL_DEBIAS, KAL_DEBIAS_DU, KAL, KAL_XAI, RANDOM, UNCERTAINTY, RandomSampling, KAL_DU
from kal.knowledge.xor import XORLoss, steep_sigmoid
from kal.metrics import F1
from kal.network import MLP, train_loop, evaluate, predict_dropout
from kal.utils import visualize_data_predictions, set_seed, check_bias_in_exp

plt.rc('animation', html='jshtml')
plt.close('all')

dataset_name = "iris_biased"
model_folder = os.path.join("models", dataset_name)
result_folder = os.path.join("results", dataset_name)
image_folder = os.path.join("images", dataset_name)
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)
if not os.path.isdir(image_folder):
    os.makedirs(image_folder)

set_seed(0)

sns.set_theme(style="whitegrid", font="Times New Roman")
now = str(datetime.datetime.now()).replace(":", ".")
# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dev = torch.device("cpu")
print(f"Working on {dev}")

strategies = [KAL_DU, KAL_DEBIAS, KAL_DEBIAS_DU, RANDOM, UNCERTAINTY]
# strategies = [ KAL_DEBIAS_DU]

print(f"Strategies: {strategies}")
# %% md

#### Loading data for the IRIS dataset

# %%

KLoss = IrisLoss
load = False
first_points = 5
n_points = 5
rand_points = 2
n_iterations = (50 - first_points) // n_points
hidden_size = 100
seeds = 10
lr = 3 * 1e-3
epochs = 200
discretize_feats = False
feature_names = load_iris().feature_names
bias = ""
xai_model = XAI_TREE(discretize_feats=discretize_feats,
                     class_names=feature_names, dev=dev)
c_loss = Expl_2_Loss(feature_names, expl=[bias], uncertainty=False, double_imp=True)


iris_dataset = load_iris()
X = iris_dataset.data
Y = iris_dataset.target
class_names = iris_dataset.target_names
print("Class names", class_names, "Feat names", feature_names)

x = MinMaxScaler().fit_transform(X)
y = OneHotEncoder(sparse=False).fit_transform(Y.reshape(-1, 1))
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

### CREATING A BIASED DATASET: adding two complete noise features


tot_points = x.shape[0]
input_size = x.shape[1]
n_classes = y.shape[1]

print("n_points", n_points, "n_iterations", n_iterations)


#%% md

#### Visualizing iris data

#%%
sns.scatterplot(x=x[:, 2], y=x[:, 3], hue=Y)
plt.savefig(f"{image_folder}\\data_labelling.png")
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.show()

# %% md
#### Defining constraints as product t-norm of the FOL rule expressing the XOR
# %%
k_loss = KLoss()(y_t, x=x_t)
sns.scatterplot(x=x_t[:, 2].numpy(), y=x_t[:, 3].numpy(), hue=k_loss.numpy())
plt.show()

# %%md
#### Calculating the prediction of the rule
# %%

def calculate_rule_prediction(x_continue: torch.Tensor) -> torch.Tensor:
    petal_length = steep_sigmoid(x_continue[:, 2], k=100, b=0.3).float()
    petal_width = steep_sigmoid(x_continue[:, 3], k=100, b=0.6).float()
    f1 = 1 - petal_length
    f2 = petal_length * (1 - petal_width)
    f3 = petal_length * petal_width
    f = torch.stack((f1, f2, f3), dim=1)
    f = torch.softmax(f, dim=1)
    return f

pred_rule = calculate_rule_prediction(x_t)
print("Rule Accuracy:", f1_score(y_t, pred_rule > 0.5, average="macro") * 100)
# sns.scatterplot(x=x_t[:, 2].numpy(), y=x_t[:, 3].numpy(), hue=pred_rule.argmax(dim=1),
#                 style=y_t.argmax(dim=1) == pred_rule.argmax(dim=1))
# plt.show()


#### Active Learning Strategy Comparison
dfs = []
skf = StratifiedKFold(n_splits=seeds)

for seed, (train_idx, test_idx) in enumerate(skf.split(x_t, y_t.argmax(dim=1))):
    train_sample = len(train_idx)
    set_seed(seed)
    first_idx = np.random.choice(train_sample, first_points, replace=False).tolist()
    print("First idx", first_idx)

    for strategy in strategies:
        active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss,
                                                        main_classes=[0],
                                                        rand_points=rand_points,
                                                        hidden_size=hidden_size,
                                                        dev=dev, cv=False,
                                                        class_names=feat_names,
                                                        mutual_excl=False, double_imp=True,
                                                        discretize_feats=discretize_feats)
        df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                              f"{seed}_seed_{strategy}_strategy.pkl")
        if os.path.exists(df_file) and load:
            df = pd.read_pickle(df_file)
            dfs.append(df)
            if "Accuracy" in df.columns:
                df['Test Accuracy'] = df['Accuracy']
            auc = df['Test Accuracy'].mean()
            print(f"Already trained {df_file}, auc: {auc}")
            continue

        df = {
            "Strategy": [],
            "Seed": [],
            "Iteration": [],
            "Active Idx": [],
            "Used Idx": [],
            "Predictions": [],
            "Train Accuracy": [],
            "Test Accuracy": [],
            "Supervision Loss": [],
            "Active Loss": [],
            "Time": [],
            "Train Idx": [],
            "Test Idx": []
        }

        x_train, y_train = x_t[train_idx], y_t[train_idx]
        x_test, y_test = x_t[test_idx], y_t[test_idx]
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        metric = F1()

        set_seed(0)
        net = MLP(n_classes, input_size, hidden_size, dropout=True).to(dev)

        # first training with few randomly selected data
        losses = []
        used_idx = first_idx.copy()
        for it in (pbar := tqdm.trange(n_iterations)):

            losses += train_loop(net, train_dataset, used_idx, epochs,
                                 lr=lr, loss=loss, device=dev)

            t = time.time()
            train_accuracy, _, preds_t = evaluate(net, train_dataset, loss=loss, device=dev,
                                                  return_preds=True, labelled_idx=used_idx)
            pred_time = time.time() - t
            print(f"Pred time {pred_time:2f}")

            if strategy in DROPOUTS:
                t = time.time()
                preds_dropout = predict_dropout(net, train_dataset)
                assert (preds_dropout - preds_t).abs().sum() > .1, \
                    "Error in computing dropout predictions"
                pred_time = time.time() - t
                print(f"Dropout time {pred_time:2f}")
            else:
                preds_dropout = None

            test_accuracy, sup_loss = evaluate(net, test_dataset, metric=metric,
                                               loss=torch.nn.BCEWithLogitsLoss(reduction="none"))
            t = time.time()
            active_idx, active_loss = active_strategy.selection(preds_t, used_idx,
                                                                n_points, x=x_train,
                                                                labels=y_train,
                                                                preds_dropout=preds_dropout,
                                                                clf=net, dataset=train_dataset)

            used_time = time.time() - t
            used_idx += active_idx

            df["Strategy"].append(strategy)
            df["Seed"].append(seed)
            df["Iteration"].append(it)
            df["Active Idx"].append(active_idx.copy())
            df["Used Idx"].append(used_idx.copy())
            df["Predictions"].append(preds_t.cpu().numpy())
            df['Train Accuracy'].append(train_accuracy)
            df["Test Accuracy"].append(test_accuracy)
            df["Supervision Loss"].append(sup_loss)
            df["Active Loss"].append(active_loss.cpu().numpy())
            df["Time"].append(used_time)
            df["Train Idx"].append(train_idx)
            df["Test Idx"].append(test_idx)

            assert isinstance(used_idx, list), "Error"

            pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                 f"train auc: {np.mean(df['Train Accuracy']):.2f}, "
                                 f"test auc: {np.mean(df['Test Accuracy']):.2f}, "
                                 f"s_l: {mean(sup_loss):.2f}, "
                                 f"l: {losses[-1]:.2f}, p: {len(used_idx)}")

        if seed == 0:
            sns.lineplot(data=losses)
            plt.yscale("log")
            plt.ylabel("Loss")
            plt.xlabel("Epochs")
            plt.title(f"Training loss variations for {strategy} "
                      f"active learning strategy")
            plt.show()

        df = pd.DataFrame(df)
        df.to_pickle(df_file)
        dfs.append(df)

dfs = pd.concat(dfs)
# dfs.to_pickle(f"{result_folder}\\metrics_{n_points}_points_{now}.pkl")
dfs.to_pickle(f"{result_folder}\\results.pkl")

mean_auc = dfs.groupby("Strategy").mean().round(2)['Test Accuracy']
std_auc = dfs.groupby(["Strategy", "Seed"]).mean().groupby("Strategy").std().round(2)['Test Accuracy']
print("AUC", mean_auc, "+-", std_auc)

# %% md

#### Displaying some pictures to visualize training

# %%
dfs = pd.read_pickle(f"{result_folder}\\results.pkl")

sns.set(style="ticks", font="Times New Roman", font_scale=1.3,
        rc={'figure.figsize': (6, 5)})
for strategy in strategies:
    # iterations = [10] if strategy != SUPERVISED else [15]
    iterations = [*range(0, 10)]
    for i in iterations:
        print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
        png_file = os.path.join(f"{image_folder}", f"{strategy}_{i}.png")
        # if not os.path.exists(png_file):
        visualize_data_predictions(x_t, i, strategy, dfs, png_file,
                                   dimensions=[2, 3], dataset="iris", seed=0)

