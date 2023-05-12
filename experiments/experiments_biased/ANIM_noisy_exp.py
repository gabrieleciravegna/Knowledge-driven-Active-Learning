#
# if __name__ == "__main__":

# %% md

# Knowledge-Driven Active Learning - Experiment on the XOR problem

# %% md
import math
import os
from functools import partial

import sklearn.model_selection

from kal.knowledge import AnimalLoss

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sympy
import torchvision
import datetime
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import torch
from torchvision.transforms import transforms
from torch.utils.data import TensorDataset, DataLoader

from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, ENTROPY_D, ENTROPY, ADV_DEEPFOOL, ADV_BIM, BALD, \
    DROPOUTS, KAL, KAL_XAI, RANDOM, UNCERTAINTY, RandomSampling, KAL_DU, KAL_DEBIAS, KAL_DEBIAS_DU
from kal.knowledge.xor import XORLoss, steep_sigmoid
from kal.metrics import F1
from kal.network import MLP, train_loop, evaluate, predict_dropout
from kal.utils import visualize_data_predictions, set_seed, check_bias_in_exp
from data.Animals import classes, CLASS_1_HOTS
from kal.knowledge.expl_to_loss import Expl_2_Loss, Expl_2_Loss_CV
from kal.xai import XAI_TREE

plt.rc('animation', html='jshtml')
plt.close('all')

dataset_name = "animals_noisy"
model_folder = os.path.join("models", dataset_name)
result_folder = os.path.join("results", dataset_name)
image_folder = os.path.join("images", dataset_name)
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)
if not os.path.isdir(image_folder):
    os.makedirs(image_folder)
data_folder = os.path.join("..", "..", "data", "Animals")
assert os.path.exists(data_folder), f"Unable to locate {data_folder}"

set_seed(0)

sns.set_theme(style="whitegrid", font="Times New Roman")
sns.set_palette(sns.color_palette()[:3])
now = str(datetime.datetime.now()).replace(":", ".")
# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dev = torch.device("cpu")
print(f"Working on {dev}")

strategies = [KAL_DU, KAL_DEBIAS, KAL_DEBIAS_DU, RANDOM, UNCERTAINTY]

# %% md
#### Generating and visualizing data for the xor problem
# %%

# %%
first_points = 100
n_points = 1000
rand_points = 0
n_iterations = (1250 - first_points) // n_points
seeds = 5
hidden_size = 100
lr = 1e-3
epochs = 250
noisy_percentage = 0.9
main_classes = range(7)
attribute_classes = range(7, 33)
discretize_feats = True
metric = F1()
height = None
load = True
print("Rand points", rand_points)

# The train dataset is composed of 19/20 of biased data (falling in the left quadrants)
# and only 1/20 of regular data

# %% md

#### Loading data for the animal's problem.
# Data pass through a RESNET 50 first which extract the data features

# %%
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
dataset = torchvision.datasets.ImageFolder(data_folder, transform=transform)

feature_extractor = torchvision.models.resnet50(pretrained=True)
feature_extractor.fc = torch.nn.Identity()
data_loader = DataLoader(dataset, batch_size=128, num_workers=8)
tot_points = len(dataset)
class_names = classes
n_classes = len(class_names)

feature_file = os.path.join(data_folder, "ResNet50-TL-feats.pth")
if os.path.isfile(feature_file):
    x = torch.load(feature_file, map_location=dev)
    y = dataset.targets
    y_multi = [CLASS_1_HOTS[dataset.classes[t]] for t in dataset.targets]
    print("Features loaded")
else:
    x, y = [], []
    with torch.no_grad():
        feature_extractor.eval(), feature_extractor.to(dev)
        for i, (batch_data, batch_labels) in enumerate(tqdm.tqdm(data_loader)):
            batch_x = feature_extractor(batch_data.to(dev))
            x.append(batch_x)
            y.append(batch_labels)
        x = torch.cat(x)
        y = torch.cat(y)
        y_multi = [CLASS_1_HOTS[dataset.classes[t]] for t in dataset.targets]
        torch.save(x, feature_file)
input_size = x.shape[1]

# %%
#### Visualizing and checking knowledge loss on the labels
KLoss = partial(AnimalLoss, names=classes)
x_t = torch.as_tensor(x, dtype=torch.float).to(dev)
y_t = torch.as_tensor(y_multi, dtype=torch.float).to(dev)
x_train, x_test, y_train, y_test = sklearn.model_selection.\
    train_test_split(x_t, y_t, test_size=int(0.25*len(y_t)))
cons_loss = KLoss()(y_t).sort()[0].cpu().numpy()
sns.scatterplot(x=[*range(len(cons_loss))], y=cons_loss)
plt.show()

penguin_class = class_names.index("PENGUIN")
penguin_idx = torch.where(y_train[:, penguin_class])[0]
fly_class = class_names.index("FLY")
penguin_attributes = torch.where(y_train[penguin_idx[0]])[0][1:]

noisy_labels_num = int(len(penguin_idx)*(noisy_percentage))
noisy_idx = np.random.choice(penguin_idx, noisy_labels_num, replace=False)
assert y_train[penguin_idx, fly_class].sum() == 0, "Error in loading data"
y_train[noisy_idx, fly_class] = 1
for penguin_attribute in penguin_attributes:
    y_train[noisy_idx, penguin_attribute] = 0
print(f"Noisy labels: {noisy_labels_num}. Clean labels: {(y_train[penguin_idx, fly_class]==0).sum()}")
assert y_train[penguin_idx, fly_class].sum() == math.floor(len(penguin_idx)*(noisy_percentage))

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_sample = len(train_dataset)
test_sample = len(test_dataset)

#### Visualizing and checking bias loss on the biased labels
bias = [""] * n_classes
bias[penguin_class] = "FLY"
xai_model = XAI_TREE(discretize_feats=True,
                     class_names=class_names, dev=dev)
c_loss = Expl_2_Loss_CV(class_names, expl=bias, uncertainty=False,
                        main_classes=attribute_classes, attribute_classes=main_classes,
                        double_imp=False)

penguin_label_train = y_train[penguin_idx]
penguin_idx_test = torch.where(y_test[:, penguin_class])[0]
penguin_label_test = y_test[penguin_idx_test]
bias_measure_train = 1 - c_loss(penguin_label_train)
bias_measure_test = 1 - c_loss(penguin_label_test)
print(f"Mean Bias in the training data: {bias_measure_train.mean():.2f}, "
      f"test {bias_measure_test.mean():.2f}")
sns.scatterplot(x=penguin_label_test[:, penguin_class].cpu(),
                y=penguin_label_test[:, fly_class].cpu(),
                hue=bias_measure_test.cpu(), legend=True).set_title("Clean Labelling")
plt.show()
sns.scatterplot(x=penguin_label_train[:, penguin_class].cpu(),
                y=penguin_label_train[:, fly_class].cpu(),
                hue=bias_measure_train.cpu(), legend=True).set_title("Noisy Labelling")
plt.show()

penguin_expl_train = xai_model.\
    explain_cv_multi_class(n_classes, y_train, [i for i in range(train_sample)])[penguin_class]
penguin_expl_test = xai_model.\
    explain_cv_multi_class(n_classes, y_test, [i for i in range(test_sample)])[penguin_class]
print(f"Train: Penguin <-> {penguin_expl_train}, bias in train set: {bias[penguin_class] in penguin_expl_train}\n"
      f"Test: Penguin <-> {penguin_expl_test}, bias in test set: {bias[penguin_class] in penguin_expl_test}")

# %%md
#### Active Learning Strategy Comparison
# %%

dfs = []
for seed in range(seeds):

    set_seed(seed)

    first_idx = RandomSampling().selection(torch.ones_like(y_train), [], first_points)[0]

    for strategy in strategies:
        active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss,
                                                        main_classes=[0],
                                                        rand_points=rand_points,
                                                        hidden_size=hidden_size,
                                                        dev=dev, cv=False,
                                                        class_names=class_names,
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
            bias_loss = c_loss(preds_train[penguin_idx])
            bias_measure = 1 - c_loss(preds_test[penguin_idx_test]).mean().item()
            bias_measure = ((bias_measure - bias_measure_test.mean()) /
                            (bias_measure_train.mean() - bias_measure_test.mean())).item()  # Normalized
            expl_formulas = xai_model.explain_cv_multi_class(n_classes, preds_test, [1 for _ in range(test_sample)])
            biased_model = bias[penguin_class] in expl_formulas
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
