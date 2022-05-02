from statistics import mean

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor

from kal.utils import visualize_data_predictions, set_seed

if __name__ == "__main__":

    # %% md

    # Knowledge-Driven Active Learning - Experiment on the XOR problem

    # %% md
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import datetime
    import random
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import tqdm
    from torch.utils.data import TensorDataset

    from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, ENTROPY_D, ENTROPY, ADV_DEEPFOOL, ADV_BIM, BALD, \
    KAL_PLUS, KALS, UNCERTAINTY_D, MARGIN_D, DROPOUTS
    from kal.knowledge.xor import XORLoss, steep_sigmoid
    from kal.metrics import MultiLabelAccuracy, F1
    from kal.network import MLP, train_loop, evaluate, predict_dropout, predict

    plt.rc('animation', html='jshtml')
    plt.close('all')

    dataset_name = "xor"
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
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Working on {dev}")

    KLoss = XORLoss
    strategies = STRATEGIES
    # strategies = KALS

    # %% md
    #### Generating and visualizing data for the xor problem
    # %%

    load = False
    tot_points = 100000
    first_points = 10
    n_points = 5
    n_iterations = (400 - first_points) // n_points
    input_size = 2
    hidden_size = 200
    seeds = 10
    lr = 1e-3
    epochs = 200

    x_t = torch.rand(tot_points, input_size).to(dev)
    y_t = (((x_t[:, 0] > 0.5) & (x_t[:, 1] < 0.5)) |
           ((x_t[:, 1] > 0.5) & (x_t[:, 0] < 0.5))
           ).float().to(dev)
    y_multi_t = torch.stack((y_t, 1 - y_t), dim=1)

    # sns.scatterplot(x=x_t[:, 0].numpy(), y=x_t[:, 1].numpy(), hue=y_t.numpy())
    # plt.savefig(f"{image_folder}\\data_labelling.png")
    # plt.show()

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

    discrete_x = steep_sigmoid(x_t, k=10).float()
    x1 = discrete_x[:, 0]
    x2 = discrete_x[:, 1]
    pred_rule = (x1 * (1 - x2)) + (x2 * (1 - x1))
    print("Rule Accuracy:",
          (pred_rule > 0.5).eq(y_t).sum().item() / y_t.shape[0] * 100)
    # sns.scatterplot(x=x_t[:, 0].numpy(), y=x_t[:, 1].numpy(), hue=pred_rule)
    # plt.show()

    #### Active Learning Strategy Comparison
    dfs = []
    skf = StratifiedKFold(n_splits=seeds)

    for seed, (train_idx, test_idx) in enumerate(skf.split(x_t, y_t)):
        train_sample = len(train_idx)
        set_seed(seed)
        first_idx = np.random.choice(train_sample, first_points, replace=False).tolist()
        print("First idx", first_idx)

        for strategy in strategies:
            active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss)
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load:
                df = pd.read_pickle(df_file)
                dfs.append(df)
                auc = df['Accuracy'].mean()
                print(f"Already trained {df_file}, auc: {auc}")
                continue

            df = {
                "Strategy": [],
                "Seed": [],
                "Iteration": [],
                "Active Idx": [],
                "Used Idx": [],
                "Predictions": [],
                "Accuracy": [],
                "Supervision Loss": [],
                "Active Loss": [],
                "Time": [],
                "Train Idx": [],
                "Test Idx": []
            }

            if strategy in [ADV_DEEPFOOL, ADV_BIM, ENTROPY, ENTROPY_D, BALD]:
                num_classes = 2
                loss = torch.nn.CrossEntropyLoss(reduction="none")
                x_train, y_train = x_t[train_idx], y_multi_t[train_idx]
                x_test, y_test = x_t[test_idx], y_multi_t[test_idx]

            else:
                num_classes = 1
                x_train, y_train = x_t[train_idx], y_t[train_idx]
                x_test, y_test = x_t[test_idx], y_t[test_idx]
                loss = torch.nn.BCEWithLogitsLoss(reduction="none")

            if strategy in DROPOUTS:
                dropout = True
            else:
                dropout = False

            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
            loss = torch.nn.BCEWithLogitsLoss(reduction="none")
            metric = F1()

            set_seed(0)
            net = MLP(input_size=input_size, hidden_size=hidden_size,
                      n_classes=num_classes, dropout=dropout).to(dev)

            # first training with few randomly selected data
            used_idx = first_idx.copy()
            losses = train_loop(net, train_dataset, used_idx, epochs,
                                lr=lr, loss=loss, visualize_loss=False)
            test_accuracy, sup_loss = evaluate(net, test_dataset, metric=metric, loss=loss)

            for it in (pbar := tqdm.trange(1, n_iterations + 1)):
                pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                     f"acc: {np.mean([0] + df['Accuracy']):.2f}, "
                                     f"s_l: {mean(sup_loss):.2f}, "
                                     f"l: {losses[-1]:.2f}, p: {len(used_idx)}")
                t = time.time()

                preds_t = predict(net, train_dataset)
                preds_dropout = predict_dropout(net, train_dataset)

                assert not dropout or (preds_dropout - preds_t).abs().sum() > .1, \
                    "Error in computing dropout predictions"

                test_accuracy, sup_loss = evaluate(net, test_dataset, metric=metric, loss=loss)

                active_idx, active_loss = active_strategy.selection(preds_t, used_idx,
                                                                    n_points, x=x_t[train_idx],
                                                                    labels=y_t[train_idx],
                                                                    preds_dropout=preds_dropout,
                                                                    clf=net, dataset=train_dataset)
                used_idx += active_idx

                df["Strategy"].append(strategy)
                df["Seed"].append(seed)
                df["Iteration"].append(it)
                df["Active Idx"].append(active_idx.copy())
                df["Used Idx"].append(used_idx.copy())
                df["Predictions"].append(preds_t.cpu().numpy())
                df["Accuracy"].append(test_accuracy)
                df["Supervision Loss"].append(sup_loss)
                df["Active Loss"].append(active_loss.cpu().numpy())
                df["Time"].append((time.time() - t))
                df["Train Idx"].append(train_idx)
                df["Test Idx"].append(test_idx)

                assert isinstance(used_idx, list), "Error"

                if it != n_iterations:
                    losses += train_loop(net, train_dataset, used_idx, epochs,
                                         lr=lr, loss=loss)
                else:
                    pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                         f"acc: {np.mean([0] + df['Accuracy']):.2f}, "
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
    dfs.to_pickle(f"{result_folder}\\metrics_{n_points}_points_{now}.pkl")

    # %%

    dfs = pd.read_pickle(os.path.join(f"{result_folder}",
                                      f"metrics_{n_points}_points_{now}.pkl"))
    dfs['Points'] = [len(used) for used in dfs['Used Idx']]
    ours = [False if "KAL" in strategy else True for strategy in dfs['Strategy']]
    dfs['Ours'] = ours

    dfs = dfs.sort_values(['Strategy', 'Seed', 'Iteration'])
    dfs = dfs.reset_index()

    rows = []
    for i, row in dfs.iterrows():
        if row['Points'] > (n_points * n_iterations + first_points):
            dfs = dfs.drop(i)

    dfs_auc = dfs.groupby("Strategy").mean()['Accuracy']
    print(dfs_auc.to_latex())
    with open(os.path.join(f"{result_folder}",
                           f"auc_latex_{now}.txt"), "w") as f:
        f.write(dfs_auc.to_latex())

    dfs_time = dfs.groupby("Strategy").mean()['Time']
    print(dfs_time.to_latex())
    with open(os.path.join(f"{result_folder}",
                           f"time_latex_{now}.txt"), "w") as f:
        f.write(dfs_time.to_latex())


    # %%

    sns.set(style="whitegrid", font_scale=1.5,
            rc={'figure.figsize': (10, 8)})
    sns.lineplot(data=dfs, x="Points", y="Accuracy",
                 hue="Strategy", style="Ours", size="Ours", legend=False, ci=None)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.ylabel("Accuracy")
    plt.xlabel("Number of points used")
    # plt.xlim([-10, 400])
    # plt.title("Comparison of the accuracies in the various strategy
    #            in function of the iterations")
    plt.legend(title='Strategy', loc='lower right', labels=sorted(strategies))
    plt.savefig(f"{image_folder}\\Accuracy_{n_points}_points_{now}.png",
                dpi=200)
    plt.show()


    # %% md

    #### Displaying some pictures to visualize training

    # %%

    # sns.set(style="ticks", font="Times New Roman", font_scale=1.3,
    #         rc={'figure.figsize': (6, 5)})
    # for strategy in strategies:
    #     # iterations = [10] if strategy != SUPERVISED else [15]
    #     iterations = [*range(1, 10)]
    #     for i in iterations:
    #         print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
    #         png_file = os.path.join(f"{image_folder}", f"{strategy}_{i}.png")
    #         # if not os.path.exists(png_file):
    #         visualize_data_predictions(x_t, i, strategy, dfs, png_file)
    #
