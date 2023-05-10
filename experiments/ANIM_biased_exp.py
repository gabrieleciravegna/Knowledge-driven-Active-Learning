#
# if __name__ == "__main__":

# %% md

# Knowledge-Driven Active Learning - Experiment on the XOR problem

# %% md
import os

import sympy

from kal.knowledge.expl_to_loss import Expl_2_Loss
from kal.xai import XAI_TREE

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils.data import TensorDataset

from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, ENTROPY_D, ENTROPY, ADV_DEEPFOOL, ADV_BIM, BALD, \
    DROPOUTS, KAL_XAI_AVOID_FOUND_BIAS, \
    KAL_XAI_AVOID_KNOWN_BIAS, KAL, KAL_XAI, RANDOM, UNCERTAINTY, RandomSampling
from kal.knowledge.xor import XORLoss, steep_sigmoid
from kal.metrics import F1
from kal.network import MLP, train_loop, evaluate, predict_dropout
from kal.utils import visualize_data_predictions, set_seed, check_bias_in_exp

plt.rc('animation', html='jshtml')
plt.close('all')

dataset_name = "xor_biased"
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

# strategies = STRATEGIES
strategies = [KAL, KAL_XAI_AVOID_KNOWN_BIAS, KAL_XAI_AVOID_FOUND_BIAS, KAL_XAI, RANDOM, UNCERTAINTY]
strategies = [RANDOM, UNCERTAINTY]

# %% md
#### Generating and visualizing data for the xor problem
# %%

KLoss = XORLoss
load = False
tot_points = 100000
first_points = 100
n_points = 5
rand_points = 0
n_iterations = (150 - first_points) // n_points
input_size = 2
hidden_size = 200
seeds = 10
lr = 1e-3
epochs = 250
discretize_feats = True
height = None
feature_names = ["x0", "x1"]
# bias = "x1 & ~x0"
bias = "x1"
xai_model = XAI_TREE(discretize_feats=discretize_feats,
                     height=height, class_names=feature_names, dev=dev)
c_loss = Expl_2_Loss(feature_names, expl=[bias], uncertainty=False, double_imp=True)
print(f"Number of random points: {rand_points}")

# The train dataset is composed of 19/20 of biased data (falling in the left quadrants)
# and only 1/20 of regular data
x_train_reg = torch.rand(tot_points//20, input_size).to(dev)
x_train_biased = torch.rand(tot_points//20*19, input_size)
x_train_biased[:, 0] = x_train_biased[:, 0]/2
x_train = torch.cat((x_train_reg, x_train_biased), dim=0)
y_train = (((x_train[:, 0] > 0.5) & (x_train[:, 1] < 0.5)) |
       ((x_train[:, 1] > 0.5) & (x_train[:, 0] < 0.5))
       ).float().to(dev)

x_test = torch.rand(tot_points//10, input_size).to(dev)
y_test = (((x_test[:, 0] > 0.5) & (x_test[:, 1] < 0.5)) |
          ((x_test[:, 1] > 0.5) & (x_test[:, 0] < 0.5))
          ).float().to(dev)
n_classes = 1

# %% md
#### Defining constraints as product t-norm of the FOL rule expressing the XOR
# %%
# preds = MLP(1, 2, 100)(x_t).detach().squeeze()
#
# k_loss = KLoss()(y_t, x=x_t)
# sns.scatterplot(x=x_t[:, 0].numpy(), y=x_t[:, 1].numpy(), hue=k_loss.numpy())
# plt.show()
#
# k_loss = KLoss(uncertainty=True)(preds, x=x_t)
# sns.scatterplot(x=x_t[:, 0].numpy(), y=x_t[:, 1].numpy(), hue=k_loss.numpy())
# plt.show()
#
# s_loss = torch.nn.BCELoss(reduction="none")(preds, y_t)
# sns.scatterplot(x=x_t[:, 0].numpy(), y=x_t[:, 1].numpy(), hue=s_loss.numpy())
# plt.show()

# %%md
#### Calculating the prediction of the rule
# %%

discrete_x = steep_sigmoid(x_test, k=10).float()
x1 = discrete_x[:, 0]
x2 = discrete_x[:, 1]
pred_rule = (x1 * (1 - x2)) + (x2 * (1 - x1))
print("Rule Accuracy:", f1_score(y_test.cpu(), pred_rule.cpu() > 0.5) * 100)
# sns.scatterplot(x=x_t[:, 0].numpy(), y=x_t[:, 1].numpy(), hue=pred_rule)
# plt.show()

# %%md
#### Active Learning Strategy Comparison
# %%

dfs = []
# skf = StratifiedKFold(n_splits=seeds)
### CREATING A BIASED DATASET AT FIRST: X1
# first_idx = np.random.choice(train_sample, first_points, replace=False).tolist()
# idx_quad_12 = torch.argwhere(x_t[:, 0] < 0.5)
# idx_quad_12 = idx_quad_12[torch.randperm(idx_quad_12.shape[0])]
# first_idx = idx_quad_12[:first_points].squeeze().cpu().numpy().tolist()
# # idx_quad_34 = torch.argwhere(x_train[:, 0] > 0.5)
# # idx_quad_34 = idx_quad_34[torch.randperm(idx_quad_34.shape[0])]
# # first_idx.extend(idx_quad_34[:first_points//50].squeeze().cpu().numpy().tolist())
# print("First idx", first_idx)
# n_classes = 1
#
# x_train, y_train = x_t[train_idx], y_t[train_idx].unsqueeze(dim=1)
# x_test, y_test = x_t[test_idx], y_t[test_idx].unsqueeze(dim=1)
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_sample = len(train_dataset)

bias_loss = 1 - c_loss(y_train, (x_train > 0.5).to(float))
bias_loss_tot = 1 - c_loss(y_train, x_train)
# assert bias_measure.sum() == 0, "Error in selecting the first index, they do not match the bias"
print(f"Mean Bias in the first training data: {bias_loss.mean():.2f}, totally {bias_loss_tot.mean():.2f}")
# sns.set_palette(sns.color_palette()[:3])
# sns.scatterplot(x=x_train[:, 0].cpu(), y=x_train[:, 1].cpu(), hue=1 - bias_loss_tot.cpu(), legend=True)
# new_idx = [0 for idx in range(len(first_idx))]
# sns.scatterplot(x=x_train[np.asarray(first_idx), 0].cpu(), y=x_train[np.asarray(first_idx), 1].cpu(),
#                 hue=new_idx, size=new_idx, sizes=[70], legend=False)
# plt.show()

# for seed, (train_idx, test_idx) in enumerate(skf.split(x_t.cpu(), y_t.cpu())):
for seed in range(seeds):

    if seed >= 5:
        break
    set_seed(seed)

    first_idx = RandomSampling().selection(torch.ones_like(y_train), [], first_points)[0]
    # sns.scatterplot(x=x_train[:, 0].cpu().numpy(), y=x_train[:, 1].cpu().numpy(), hue=y_train.squeeze().cpu().numpy())
    # plt.scatter(x=x_train[first_idx, 0].cpu().numpy(), y=x_train[first_idx, 1].cpu().numpy(), c="red")
    # plt.savefig(f"{image_folder}\\data_labelling.png")
    # plt.show()

    for strategy in strategies:
        active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss,
                                                        main_classes=[0],
                                                        rand_points=rand_points,
                                                        hidden_size=hidden_size,
                                                        dev=dev, cv=False,
                                                        class_names=feature_names,
                                                        mutual_excl=False, double_imp=True,
                                                        discretize_feats=discretize_feats)
        if first_points == 10:
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                              f"{seed}_seed_{strategy}_strategy.pkl")
        else:
            df_file = os.path.join(result_folder, f"metrics_{first_points}_fp_{n_points}_np_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
        if os.path.exists(df_file) and load:
            df = pd.read_pickle(df_file)
            dfs.append(df)
            auc, b_l, b_m = df['Test Accuracy'].mean(), df['Bias Measure'].mean(), df['Biased Model'].mean()
            print(f"Already trained {df_file}, auc: {auc:.2f}, bias: {b_l:.2f}, biased (%): {b_m:.2f}")
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
            "Bias Measure": [],
            "Biased Model": [],
            "Active Loss": [],
            "Time": [],
            # "Train Idx": [],
            # "Test Idx": []
        }

        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        metric = F1()

        set_seed(0)
        net = MLP(n_classes, input_size, hidden_size, dropout=True).to(dev)

        # first training with few randomly selected data
        losses = []
        used_idx = first_idx.copy()
        for it in (pbar := tqdm.trange(n_iterations)):

            losses += train_loop(net, train_dataset, used_idx, epochs,
                                 lr=lr, loss=loss)
            t = time.time()
            train_accuracy, _, preds_train = evaluate(net, train_dataset, loss=loss, device=dev,
                                                  return_preds=True, labelled_idx=used_idx)
            pred_time = time.time() - t
            # print(f"Pred time {pred_t:2f}")
            if strategy in DROPOUTS:
                t = time.time()
                preds_dropout = predict_dropout(net, train_dataset)
                assert (preds_dropout - preds_train).abs().sum() > .1, \
                    "Error in computing dropout predictions"
                pred_time = time.time() - t
                # print(f"Dropout time {pred_t:2f}")
            else:
                preds_dropout = None

            test_accuracy, sup_loss, preds_test = evaluate(net, test_dataset, metric=metric,
                                                           device=dev, return_preds=True,
                                                           loss=torch.nn.BCEWithLogitsLoss(reduction="none"))
            bias_measure = 1 - c_loss(preds_test, x_test.to(float), debug=False).mean().item()
            expl_formulas = xai_model.explain(x_test, preds_test, range(len(y_test)))
            biased_model = int(check_bias_in_exp(expl_formulas[0], bias))
            print(expl_formulas)

            t = time.time()

            if strategy == KAL_XAI_AVOID_KNOWN_BIAS:
                bias_strategy = True
                # formulas = ["x1"]
                formulas = [bias]
                # formulas = [str(sympy.simplify_logic(f"~(x1 & ~x0)"))]
            elif strategy == KAL_XAI_AVOID_FOUND_BIAS:
                if biased_model:
                    formulas = [bias]
                    bias_strategy = True
                    # formulas = [str(sympy.simplify_logic(f"~({expl_formulas[0]})"))]
                    # active_strategy.rand_points = 0
                else:
                    bias_strategy = False
                    # formulas = expl_formulas
                    formulas = [str(sympy.simplify_logic(f"({expl_formulas[0]})"))]
                    # active_strategy.rand_points = n_points
            else:
                formulas = None
                bias_strategy = False
            active_idx, active_loss = active_strategy.selection(preds_train, used_idx,
                                                                n_points, x=x_train,
                                                                labels=y_train.squeeze(),
                                                                formulas=formulas,
                                                                bias_strategy=bias_strategy,
                                                                preds_dropout=preds_dropout,
                                                                clf=net, dataset=train_dataset)
            used_time = time.time() - t

            used_idx += active_idx

            df["Strategy"].append(strategy)
            df["Seed"].append(seed)
            df["Iteration"].append(it)
            df["Active Idx"].append(active_idx.copy())
            df["Used Idx"].append(used_idx.copy())
            df["Predictions"].append(preds_train.cpu().numpy())
            df['Train Accuracy'].append(train_accuracy)
            df["Test Accuracy"].append(test_accuracy)
            df["Supervision Loss"].append(sup_loss)
            df["Bias Measure"].append(bias_measure)
            df["Biased Model"].append(biased_model)
            df["Active Loss"].append(active_loss.cpu().numpy())
            df["Time"].append(used_time)
            # df["Train Idx"].append(train_idx)
            # df["Test Idx"].append(test_idx)

            assert isinstance(used_idx, list), "Error"

            pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                 f"train acc: {df['Train Accuracy'][-1]:.2f}, "
                                 f"test acc: {df['Test Accuracy'][-1]:.2f}, "
                                 # f"a_l: {active_loss[-1]:.2f}, "
                                 # f"s_l: {losses[-1]:.2f}, "
                                 f"biased: {df['Biased Model'][-1]}, "
                                 f"bias: {df['Bias Measure'][-1]:.2f}, "
                                 f"p: {len(used_idx)}")

            visualize_data_predictions(x_train, it, strategy, pd.DataFrame(df), None,
                                       seed=seed)
            visualize_data_predictions(x_train, it, strategy, pd.DataFrame(df), None,
                                       seed=seed, active_loss=True)

        # if seed == 0:
        #     sns.lineplot(data=losses)
        #     plt.yscale("log")
        #     plt.ylabel("Loss")
        #     plt.xlabel("Epochs")
        #     plt.title(f"Training loss variations for {strategy} "
        #               f"active learning strategy")
        #     plt.show()

        df = pd.DataFrame(df)
        df.to_pickle(df_file)
        dfs.append(df)

        # sns.lineplot(data=df, x="Iteration", y="Bias Measure")
        # plt.title(f"Bias measure for {strategy} strategy")
        # plt.show()
        #
        # sns.lineplot(data=df, x="Iteration", y="Biased Model")
        # plt.title(f"Biased Models in iterations for {strategy} strategy")
        # plt.show()

dfs = pd.concat(dfs).reset_index()
# dfs.to_pickle(f"{result_folder}\\metrics_{n_points}_points_{now}.pkl")
dfs.to_pickle(f"{result_folder}\\results.pkl")

mean_auc = dfs.groupby("Strategy").mean().round(2)['Test Accuracy']
std_auc = dfs.groupby(["Strategy", "Seed"]).mean().groupby("Strategy").std().round(2)['Test Accuracy']
print("AUC", mean_auc, "+-", std_auc)

sns.lineplot(data=dfs, x="Iteration", y="Test Accuracy", hue="Strategy")
plt.title(f"Test Accuracy in iterations among different strategies")
plt.show()

sns.lineplot(data=dfs, x="Iteration", y="Bias Measure", hue="Strategy")
plt.title(f"Bias measure among different strategies")
plt.show()

sns.lineplot(data=dfs, x="Iteration", y="Biased Model", hue="Strategy")
plt.title(f"Biased Models in iterations among different strategies")
plt.show()
# %% md

### Displaying some pictures to visualize training

# %%
# sns.set(style="ticks", font_scale=1.8,
#         rc={'figure.figsize': (6, 5)})
# # for seed in range(seeds):
# for seed in [0]:
#     for strategy in STRATEGIES:
#         if strategy in DROPOUTS:
#             continue
#         iterations = [0, 4, 9, 17]
#         for i in iterations:
#             print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
#             png_file = os.path.join(f"{image_folder}", f"{strategy}_it_{i}_s_{seed}.png")
#             if not os.path.exists(png_file) or True:
#                 visualize_data_predictions(x_t, i, strategy, dfs, png_file,
#                                            seed=seed)
#             else:
#                 print(png_file + " Already existing")
