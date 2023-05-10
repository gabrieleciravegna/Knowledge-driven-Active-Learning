from sklearn.metrics import f1_score

if __name__ == "__main__":

    # %% md

    # Knowledge-Driven Active Learning - Experiment on the XOR problem

    # %% md
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import datetime
    import random
    import time
    from statistics import mean

    from sklearn.model_selection import StratifiedKFold

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import tqdm
    from torch.utils.data import TensorDataset

    from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, ENTROPY_D, ENTROPY, ADV_DEEPFOOL, ADV_BIM, BALD, \
    KALS, DROPOUTS, KAL_XAI_DROP_DU, KAL_XAI_DU, KAL_DU, KAL_DROP_DU, KAL, KAL_XAI
    from kal.knowledge.xor import XORLoss, steep_sigmoid
    from kal.metrics import F1
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

    # strategies = STRATEGIES
    strategies =  [KAL_XAI_DU, KAL_DU,]

    # %% md
    #### Generating and visualizing data for the xor problem
    # %%

    KLoss = XORLoss
    load = False
    tot_points = 100000
    first_points = 10
    n_points = 5
    rand_points = 0
    n_iterations = (100 - first_points) // n_points
    input_size = 2
    hidden_size = 200
    seeds = 10
    lr = 1e-3
    epochs = 250
    discretize_feats = True
    print(f"Number of random points: {rand_points}")

    x_t = torch.rand(tot_points, input_size).to(dev)
    y_t = (((x_t[:, 0] > 0.5) & (x_t[:, 1] < 0.5)) |
           ((x_t[:, 1] > 0.5) & (x_t[:, 0] < 0.5))
           ).float().to(dev)
    y_multi_t = torch.stack((1 - y_t, y_t), dim=1)
    feat_names = ["X1", "X2"]

    assert (y_multi_t.argmax(dim=1) == y_t).all(), "Error in computing y multi label"

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
    print("Rule Accuracy:", f1_score(y_t.cpu(), pred_rule.cpu() > 0.5)* 100)
    # sns.scatterplot(x=x_t[:, 0].numpy(), y=x_t[:, 1].numpy(), hue=pred_rule)
    # plt.show()

    # %%md
    #### Active Learning Strategy Comparison
    # %%

    dfs = []
    skf = StratifiedKFold(n_splits=seeds)

    for seed, (train_idx, test_idx) in enumerate(skf.split(x_t.cpu(), y_t.cpu())):
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
                                                            class_names=["x0", "x1"],
                                                            mutual_excl=False, double_imp=True,
                                                            discretize_feats=discretize_feats)
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load:
                df = pd.read_pickle(df_file)
                dfs.append(df)
                if "Test Accuracy" in df.columns:
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

            if strategy in [ADV_DEEPFOOL, ADV_BIM, ENTROPY, ENTROPY_D, BALD]:
                n_classes = 2
                x_train, y_train = x_t[train_idx], y_multi_t[train_idx]
                x_test, y_test = x_t[test_idx], y_multi_t[test_idx]
            else:
                n_classes = 1
                x_train, y_train = x_t[train_idx], y_t[train_idx].unsqueeze(dim=1)
                x_test, y_test = x_t[test_idx], y_t[test_idx].unsqueeze(dim=1)
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
                pred_t = time.time() - t
                # print(f"Pred time {pred_t:2f}")
                if strategy in DROPOUTS:
                    t = time.time()
                    preds_dropout = predict_dropout(net, train_dataset)
                    assert (preds_dropout - preds_t).abs().sum() > .1, \
                        "Error in computing dropout predictions"
                    pred_t = time.time() - t
                    # print(f"Dropout time {pred_t:2f}")
                else:
                    preds_dropout = None

                test_accuracy, sup_loss = evaluate(net, test_dataset, metric=metric, device=dev,
                                                   loss=torch.nn.BCEWithLogitsLoss(reduction="none"))
                t = time.time()
                active_idx, active_loss = active_strategy.selection(preds_t, used_idx,
                                                                    n_points, x=x_train,
                                                                    labels=y_train,
                                                                    preds_dropout=preds_dropout,
                                                                    clf=net, dataset=train_dataset,
                                                                    # formulas=["x0 & ~x1 | x1 & ~x0"]
                                                                    )
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

                # visualize_data_predictions(x_t, it, strategy, pd.DataFrame(df), None,
                #                            seed=seed)
                # visualize_data_predictions(x_t, it, strategy, pd.DataFrame(df), None,
                #                            seed=seed, active_loss=True)

                assert isinstance(used_idx, list), "Error"

                pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                     f"train auc: {np.mean(df['Train Accuracy']):.2f}, "
                                     f"test auc: {np.mean(df['Test Accuracy']):.2f}, "
                                     f"a_l: {active_loss.mean().item():.2f}, "
                                     f"l: {losses[-1]:.2f}, p: {len(used_idx)}")
                print()

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

    ### Displaying some pictures to visualize training

    # %%

    sns.set(style="ticks", font_scale=1.8,
            rc={'figure.figsize': (6, 5)})
    # for seed in range(seeds):
    for seed in [4]:
        for strategy in STRATEGIES:
            if strategy in DROPOUTS:
                continue
            iterations = [0, 4, 9, 17]
            for i in iterations:
                print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
                png_file = os.path.join(f"{image_folder}", f"{strategy}_it_{i}_s_{seed}.png")
                if not os.path.exists(png_file) or True:
                    visualize_data_predictions(x_t, i, strategy, dfs, png_file,
                                               seed=seed)
                else:
                    print(png_file + " Already existing")
