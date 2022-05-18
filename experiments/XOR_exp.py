from statistics import mean

import pandas
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor

from kal.knowledge.knowledge_loss import CombinedLoss

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
    KAL_PLUS, KALS, UNCERTAINTY_D, MARGIN_D, DROPOUTS, KCENTER, NAME_MAPPINGS, RANDOM, NAME_MAPPINGS_LATEX, BALD2, \
    KAL_DU, MARGIN, KAL_U, KAL_STAR_DROP_DU, KAL_STAR_DU
    from kal.knowledge.xor import XORLoss, steep_sigmoid
    from kal.metrics import MultiLabelAccuracy, F1
    from kal.network import MLP, train_loop, evaluate, predict_dropout, predict
    from kal.utils import visualize_data_predictions, set_seed

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
    # strategies = [BALD, ENTROPY, ENTROPY_D]
    strategies = STRATEGIES
    # strategies.pop(strategies.index(KCENTER))
    # strategies = KALS

    # %% md
    #### Generating and visualizing data for the xor problem
    # %%

    load = True
    tot_points = 100000
    first_points = 10
    n_points = 5
    n_iterations = (400 - first_points) // n_points
    input_size = 2
    hidden_size = 200
    seeds = 10
    lr = 1e-3
    epochs = 250

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

    # %%md
    #### Active Learning Strategy Comparison
    # %%

    dfs = []
    skf = StratifiedKFold(n_splits=seeds)

    for seed, (train_idx, test_idx) in enumerate(skf.split(x_t, y_t)):
        train_sample = len(train_idx)
        set_seed(seed)
        first_idx = np.random.choice(train_sample, first_points, replace=False).tolist()
        print("First idx", first_idx)

        for strategy in strategies:
            active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss, main_classes=[0, 1])
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load:
                df = pd.read_pickle(df_file)
                dfs.append(df)
                df_first_idx = df['Used Idx'][0][:first_points]
                assert df_first_idx == first_idx, \
                    f"Error in loading the data, loaded first points are differents\n{df_first_idx}"
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

            if strategy in [ADV_DEEPFOOL, ADV_BIM, ENTROPY, ENTROPY_D, BALD, BALD2]:
                num_classes = 2
                x_train, y_train = x_t[train_idx], y_multi_t[train_idx]
                x_test, y_test = x_t[test_idx], y_multi_t[test_idx]
            else:
                num_classes = 1
                x_train, y_train = x_t[train_idx], y_t[train_idx]
                x_test, y_test = x_t[test_idx], y_t[test_idx]

            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
            if strategy in [KAL_STAR_DU, KAL_STAR_DROP_DU]:
                loss = CombinedLoss(KLoss)
            else:
                loss = torch.nn.BCEWithLogitsLoss(reduction="none")
            metric = F1()

            set_seed(0)
            net = MLP(input_size=input_size, hidden_size=hidden_size,
                      n_classes=num_classes, dropout=True).to(dev)

            # first training with few randomly selected data
            losses = []
            used_idx = first_idx.copy()
            for it in (pbar := tqdm.trange(n_iterations)):
                t = time.time()

                losses += train_loop(net, train_dataset, used_idx, epochs,
                                     lr=lr, loss=loss, visualize_loss=False)
                preds_t = predict(net, train_dataset)

                if strategy in DROPOUTS:
                    preds_dropout = predict_dropout(net, train_dataset)
                    assert (preds_dropout - preds_t).abs().sum() > .1, \
                        "Error in computing dropout predictions"
                else:
                    preds_dropout = None

                test_accuracy, sup_loss = evaluate(net, test_dataset, metric=metric,
                                                   loss=torch.nn.BCEWithLogitsLoss(reduction="none")
)

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

                pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                     f"auc: {np.mean(df['Accuracy']):.2f}, "
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

    # %%

    dfs = pd.read_pickle(f"{result_folder}\\results.pkl")
    # dfs = pd.read_pickle(os.path.join(f"{result_folder}",
    #                                   f"metrics_{n_points}_points_{now}.pkl"))
    # dfs['Points'] = [len(used) for used in dfs['Used Idx']]
    # ours = [True if "KAL" in strategy else False for strategy in dfs['Strategy']]
    # dfs['Ours'] = ours
    #
    # dfs = dfs.sort_values(['Strategy', 'Seed', 'Iteration'])
    # dfs = dfs.reset_index()
    #
    # rows = []
    # Strategies = []
    # for i, row in dfs.iterrows():
    #     if row['Points'] > (n_points * n_iterations + first_points):
    #         dfs = dfs.drop(i)
    #     else:
    #         Strategies.append(NAME_MAPPINGS_LATEX[row['Strategy']])
    # dfs['Strategy'] = Strategies

    # aucs = []
    # dfs_auc_mean = dfs.groupby("Strategy").mean()['Accuracy'].tolist()
    # dfs_auc_std = dfs.groupby(["Strategy", "Seed"]).mean()['Accuracy'] \
    #     .groupby("Strategy").std().tolist()
    # for mean, std in zip(dfs_auc_mean, dfs_auc_std):
    #     auc = f"${mean:.2f}$ {{\\tiny $\\pm {std:.2f}$ }}"
    #     aucs.append(auc)
    # df_auc = pd.DataFrame({
    #     "Strategy": np.unique(Strategies),
    #     "AUC": aucs,
    # }).set_index("Strategy")
    # print(df_auc.to_latex(float_format="%.2f", escape=False))
    # with open(os.path.join(f"{result_folder}",
    #                        f"table_auc_latex_{now}.txt"), "w") as f:
    #     f.write(df_auc.to_latex(float_format="%.2f", escape=False))
    #
    # final_accs = []
    # dfs_final_accs = dfs[dfs['Iteration'] == n_iterations - 1]
    # final_accs_mean = dfs_final_accs.groupby("Strategy").mean()['Accuracy'].tolist()
    # final_accs_std = dfs_final_accs.groupby("Strategy").std()['Accuracy'].tolist()
    # for mean, std in zip(final_accs_mean, final_accs_std):
    #     final_acc = f"${mean:.2f}$ {{\\tiny $\\pm {std:.2f}$ }}"
    #     final_accs.append(final_acc)
    # df_final_acc = pd.DataFrame({
    #     "Strategy": np.unique(Strategies),
    #     "Final F1": final_accs,
    # }).set_index("Strategy")
    # print(df_final_acc.to_latex(float_format="%.2f", escape=False))
    # with open(os.path.join(f"{result_folder}",
    #                        f"table_final_acc_latex_{now}.txt"), "w") as f:
    #     f.write(df_final_acc.to_latex(float_format="%.2f", escape=False))
    #
    # times = []
    # dfs_time_mean = dfs.groupby("Strategy").mean()['Time']
    # base_time = dfs_time_mean[dfs_time_mean.index == RANDOM].item()
    # for time in dfs_time_mean.tolist():
    #     time = time / base_time
    #     time = time if time >= 1. else 1.
    #     time = f"${time:.2f}$ x"
    #     times.append(time)
    # df_times = pd.DataFrame({
    #     "Strategy": np.unique(Strategies),
    #     "Times": times
    # }).set_index("Strategy")
    # print(df_times.to_latex(float_format="%.2f", escape=False))
    # with open(os.path.join(f"{result_folder}",
    #                        f"table_times_latex_{now}.txt"), "w") as f:
    #     f.write(df_times.to_latex(float_format="%.2f", escape=False))

    # %%
    #
    # sns.set(style="whitegrid", font_scale=1.5,
    #         rc={'figure.figsize': (10, 8)})
    # sns.lineplot(data=dfs, x="Points", y="Accuracy",
    #              hue="Strategy", style="Ours", size="Ours",
    #              legend=False, ci=None, style_order=[1, 0],
    #              size_order=[1, 0], sizes=[4, 2])
    # sns.despine(left=True, bottom=True)
    # plt.tight_layout()
    # plt.ylabel("Accuracy")
    # plt.xlabel("Number of points used")
    # # plt.xlim([-10, 400])
    # # plt.title("Comparison of the accuracies in the various strategy
    # #            in function of the iterations")
    # labels = [NAME_MAPPINGS[strategy] for strategy in sorted(strategies)]
    # plt.legend(title='Strategy', loc='lower right', labels=labels)
    # plt.savefig(f"{image_folder}\\Accuracy_{dataset_name}_{n_points}_points_{now}.png",
    #             dpi=200)
    # plt.show()


    # %% md

    ### Displaying some pictures to visualize training

    # %%

    sns.set(style="ticks", font_scale=1.8,
            rc={'figure.figsize': (6, 5)})
    # for seed in range(seeds):
    for seed in [4]:
        for strategy in [KCENTER]:
        # for strategy in [KAL_DU]:
            iterations = [19]
            # iterations = [0, 4, 9]
            for i in iterations:
                print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
                png_file = os.path.join(f"{image_folder}", f"{strategy}_it_{i}_s_{seed}.png")
                # if not os.path.exists(png_file):
                visualize_data_predictions(x_t, i, strategy, dfs, png_file,
                                           seed=seed)

