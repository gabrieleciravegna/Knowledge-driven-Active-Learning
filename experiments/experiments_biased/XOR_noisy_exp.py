#
# if __name__ == "__main__":

# %% md

# Knowledge-Driven Active Learning - Experiment on the XOR problem

# %% md
import math
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
    DROPOUTS, KAL_DEBIAS, KAL_DEBIAS_DU, KAL, KAL_XAI, RANDOM, UNCERTAINTY, RandomSampling, KAL_DU
from kal.knowledge.xor import XORLoss, steep_sigmoid
from kal.metrics import F1
from kal.network import MLP, train_loop, evaluate, predict_dropout
from kal.utils import visualize_data_predictions, set_seed, check_bias_in_exp

plt.rc('animation', html='jshtml')
plt.close('all')

dataset_name = "xor_noisy"
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
# strategies = [KAL_DEBIAS_DU, KAL_DEBIAS]

# %% md
#### Generating and visualizing data for the xor problem
# %%

KLoss = XORLoss
load = False
tot_points = 100000
first_points = 100
n_points = 5
rand_points = 0
noisy_percentage = 0.9
n_iterations = (200 - first_points) // n_points
input_size = 2
hidden_size = 200
seeds = 10
lr = 1e-3
epochs = 250
discretize_feats = True
feature_names = ["x0", "x1"]
bias = "x1 & ~x0"
xai_model = XAI_TREE(discretize_feats=True,
                     class_names=feature_names, dev=dev)
c_loss = Expl_2_Loss(feature_names, expl=[bias], uncertainty=False, double_imp=True, discretize_feats=discretize_feats)

### CREATING A NOISY TRAIN DATASET: X1
# The train dataset is composed of 90% of noisy data
# and only of 10% of regular data
x_train = torch.rand(tot_points, input_size).to(dev)
y_train = (((x_train[:, 0] > 0.5) & (x_train[:, 1] < 0.5)) |
           ((x_train[:, 1] > 0.5) & (x_train[:, 0] < 0.5))
           ).float().to(dev)

right_top_idx = torch.where((x_train[:, 0] > 0.5) & (x_train[:, 1] < 0.5))[0]
noisy_labels_num = int(len(right_top_idx)*(noisy_percentage))
noisy_y_idx = np.random.choice(right_top_idx, noisy_labels_num, replace=False)
y_train[noisy_y_idx] = 0
assert y_train[right_top_idx].sum() == math.ceil(len(right_top_idx)*(1 - noisy_percentage))
print(f"Noisy labels: {noisy_labels_num}. Clean labels: {y_train[right_top_idx].sum()}")

x_test = torch.rand(tot_points//10, input_size).to(dev)
y_test = (((x_test[:, 0] > 0.5) & (x_test[:, 1] < 0.5)) |
          ((x_test[:, 1] > 0.5) & (x_test[:, 0] < 0.5))
          ).float().to(dev)
n_classes = 1


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
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_sample = len(train_dataset)

bias_measure_train = 1 - c_loss(y_train, x_train)
bias_measure_test = 1 - c_loss(y_test, x_test)
print(f"Mean Bias in the training data: {bias_measure_train.mean():.2f}, test {bias_measure_test.mean():.2f}")
sns.scatterplot(x=x_train[:, 0].cpu(), y=x_train[:, 1].cpu(), hue=y_train.cpu(), legend=True).set_title("Noisy Labelling")
plt.show()
sns.scatterplot(x=x_train[:, 0].cpu(), y=x_train[:, 1].cpu(), hue=bias_measure_train.cpu(), legend=True).set_title(f"Bias {bias} level")
plt.show()

for seed in range(seeds):

    set_seed(seed)

    first_idx = RandomSampling().selection(torch.ones_like(y_train), [], first_points)[0]

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

        df = []
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
            train_accuracy, _, preds_train = evaluate(net, train_dataset, loss=loss, device=dev,
                                                      return_preds=True, labelled_idx=used_idx)
            if strategy in DROPOUTS:
                preds_dropout = predict_dropout(net, train_dataset)
                assert (preds_dropout - preds_train).abs().sum() > .1, \
                    "Error in computing dropout predictions"
            else:
                preds_dropout = None

            test_accuracy, sup_loss, preds_test = evaluate(net, test_dataset, metric=metric,
                                                           device=dev, return_preds=True,
                                                           loss=loss)
            bias_loss = c_loss(preds_train, x_train)
            bias_measure = 1 - c_loss(preds_test, x_test).mean().item()
            bias_measure = ((bias_measure - bias_measure_test.mean()) /
                            (bias_measure_train.mean() - bias_measure_test.mean())).item()  # Normalized
            expl_formulas = xai_model.explain(x_test, preds_test, range(len(y_test)))
            biased_model = int(check_bias_in_exp(expl_formulas[0], bias))
            print(expl_formulas)
            print(f"Bias in dataset: {1 - c_loss(y_train[used_idx], x_train[used_idx]).mean():.2f}")

            t = time.time()

            active_idx, active_loss = active_strategy.selection(preds_train, used_idx,
                                                                n_points, x=x_train,
                                                                labels=y_train.squeeze(),
                                                                formulas=expl_formulas,
                                                                bias=bias,
                                                                biased_model=biased_model,
                                                                preds_dropout=preds_dropout,
                                                                clf=net, dataset=train_dataset)
            used_time = time.time() - t
            used_idx += active_idx

            df.append({
                "Strategy": strategy,
                "Seed": seed,
                "Iteration": it,
                "Active Idx": active_idx.copy(),
                "Used Idx": used_idx.copy(),
                "Predictions": preds_train.cpu().numpy(),
                'Train Accuracy': train_accuracy,
                "Test Accuracy": test_accuracy,
                "Supervision Loss": sup_loss,
                "Bias Measure": bias_measure,
                "Biased Model": biased_model,
                "Bias Loss": bias_loss.cpu().numpy(),
                "Active Loss": active_loss.cpu().numpy(),
                "Time": used_time,
            })

            assert isinstance(used_idx, list), "Error"

            pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                 f"train acc: {train_accuracy:.2f}, "
                                 f"test acc: {test_accuracy:.2f}, "
                                 f"biased: {biased_model}, "
                                 f"bias: {bias_measure:.2f}, "
                                 f"p: {len(used_idx)}")

            if (it == 0 or it == n_iterations - 1) and seed == 0:
                visualize_data_predictions(x_train, it, strategy, pd.DataFrame(df), None,
                                           seed=seed)
                visualize_data_predictions(x_train, it, strategy, pd.DataFrame(df), None,
                                           seed=seed, bias=True)
                sns.scatterplot(x=x_train[used_idx, 0], y=x_train[used_idx, 1], hue=y_train[used_idx])
                plt.show()

        if seed == 0:
            sns.lineplot(data=losses)
            plt.yscale("log")
            plt.ylabel("Loss")
            plt.xlabel("Epochs")
            plt.title(f"Training loss variations for {strategy} "
                      f"active learning strategy")
            plt.show()

        df = pd.DataFrame.from_dict(df)
        df.to_pickle(df_file)
        dfs.append(df)

dfs = pd.concat(dfs).reset_index()
# dfs.to_pickle(f"{result_folder}\\metrics_{n_points}_points_{now}.pkl")
dfs.to_pickle(f"{result_folder}\\results.pkl")

dfs['# points'] = dfs['Iteration']*n_points + first_points
mean_auc = dfs.groupby("Strategy").mean().round(2)['Test Accuracy']
std_auc = dfs.groupby(["Strategy", "Seed"]).mean().groupby("Strategy").std().round(2)['Test Accuracy']
print("AUC", mean_auc, "+-", std_auc)

sns.lineplot(data=dfs, x='# points', y="Test Accuracy", hue="Strategy")
# plt.gca().invert_yaxis()
plt.title(f"Test Accuracy in iterations among different strategies")
plt.savefig(os.path.join(image_folder, f"Test Accuracy {strategies}"))
plt.show()

dfs['Bias Measure'] = dfs['Bias Measure'].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
sns.lineplot(data=dfs, x='# points', y="Bias Measure", hue="Strategy")
plt.title(f"Bias measure among different strategies")
plt.savefig(os.path.join(image_folder, f"Bias Measure {strategies}"))
plt.show()

sns.lineplot(data=dfs, x='# points', y="Biased Model", hue="Strategy")
plt.title(f"Biased Models in iterations among different strategies")
plt.savefig(os.path.join(image_folder, f"Biased Model {strategies}"))
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
