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

    from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, ENTROPY_D, ENTROPY, ADV_DEEPFOOL, ADV_BIM, BALD
    from kal.knowledge.xor import XORLoss, steep_sigmoid
    from kal.metrics import MultiLabelAccuracy
    from kal.network import MLP, train_loop, evaluate, predict_dropout

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
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    sns.set_theme(style="whitegrid", font="Times New Roman")
    now = str(datetime.datetime.now()).replace(":", ".")
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Working on {dev}")

    KLoss = XORLoss
    strategies = STRATEGIES

    # %% md
    #### Generating and visualizing data for the xor problem
    # %%

    tot_points = 10000
    first_points = 10
    n_points = 5
    n_iterations = 98
    seeds = range(10)
    x_t = torch.rand(tot_points, 2).to(dev)
    y_t = (((x_t[:, 0] > 0.5) & (x_t[:, 1] < 0.5)) |
           ((x_t[:, 1] > 0.5) & (x_t[:, 0] < 0.5))
           ).float().to(dev)
    y_multi_t = torch.stack((y_t, 1 - y_t), dim=1)

    dataset = TensorDataset(x_t, y_t)
    dataset_multi = TensorDataset(x_t, y_multi_t)

    # sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y)
    # plt.savefig(f"{image_folder}\\data_labelling.png")
    # plt.show()

    # %% md
    #### Defining constraints as product t-norm of the FOL rule expressing the XOR
    # %%

    k_loss = KLoss()(y_t, x=x_t)
    # sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=k_loss)
    # plt.show()

    # %%md
    #### Calculating the prediction of the rule
    # %%

    discrete_x = steep_sigmoid(x_t, k=100).float()
    x1 = discrete_x[:, 0]
    x2 = discrete_x[:, 1]
    pred_rule = (x1 * (1 - x2)) + (x2 * (1 - x1))

    print("Rule Accuracy:",
          (pred_rule > 0.5).eq(y_t).sum().item() / y_t.shape[0] * 100)

    # sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=pred_rule)
    # plt.show()

    #### Active Learning Strategy Comparison
    load = False
    dfs = []
    first_idx = [torch.randperm(tot_points)[:first_points].numpy().tolist() for _ in seeds]

    for strategy in strategies:
        for seed in seeds:
            active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss)
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load:
                print(f"Already trained {df_file}")
                df = pd.read_pickle(df_file)
                dfs.append(df)
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
            }

            if strategy in [ADV_DEEPFOOL, ADV_BIM, ENTROPY, ENTROPY_D, BALD]:
                num_classes = 2
                train_dataset = dataset_multi
                if strategy in [ENTROPY, ENTROPY_D, BALD]:
                    dropout = True
                else:
                    dropout = False
            else:
                num_classes = 1
                train_dataset = dataset
                dropout = False

            net = MLP(input_size=2, hidden_size=100, n_classes=num_classes,
                      dropout=dropout).to(dev)
            metric = MultiLabelAccuracy(num_classes)
            loss = torch.nn.BCELoss(reduction="none")

            # first training with few randomly selected data
            used_idx = first_idx[seed].copy()
            losses = train_loop(net, train_dataset, used_idx, loss=loss, visualize_loss=True)
            accuracy, _, sup_loss = evaluate(net, train_dataset, metric=metric)

            for it in (pbar := tqdm.trange(1, n_iterations + 1)):
                pbar.set_description(f"{strategy}, seed {seed + 1}/{len(seeds)}, "
                                     f"acc: {accuracy:.2f}, l: {sup_loss.mean():.2f}, p: {len(used_idx)}")
                t = time.time()

                accuracy, preds_t, sup_loss = evaluate(net, train_dataset, loss=loss, metric=metric)
                preds_dropout = predict_dropout(net, train_dataset)

                active_idx, active_loss = active_strategy.selection(preds_t, used_idx, n_points,
                                                                    x=x_t, labels=y_t,
                                                                    preds_dropout=preds_dropout,
                                                                    clf=net, dataset=train_dataset)
                used_idx += active_idx

                df["Strategy"].append(strategy)
                df["Seed"].append(seed)
                df["Iteration"].append(it)
                df["Active Idx"].append(active_idx.copy())
                df["Used Idx"].append(used_idx.copy())
                df["Predictions"].append(preds_t.cpu().numpy())
                df["Accuracy"].append(accuracy)
                df["Supervision Loss"].append(sup_loss.cpu().numpy())
                df["Active Loss"].append(active_loss.cpu().numpy())
                df["Time"].append((time.time() - t))

                assert isinstance(used_idx, list), "Error"

                if it != n_iterations:
                    losses += train_loop(net, train_dataset, used_idx)

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
    dfs = dfs.sort_values(['Strategy'])
    dfs['Accuracy'] = [acc.item() if isinstance(acc, torch.Tensor) else acc
                       for acc in dfs['Accuracy']]
    dfs = dfs.reset_index()
    # %%

    sns.set(style="whitegrid", font_scale=1.2,
            rc={'figure.figsize': (10, 8)})
    sns.lineplot(data=dfs, x="Points", y="Accuracy",
                 hue="Strategy", ci=None)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.ylabel("Accuracy")
    plt.xlabel("Number of points used")
    plt.xlim([-10, 400])
    # plt.title("Comparison of the accuracies in the various strategy
    #            in function of the iterations")
    plt.savefig(f"{image_folder}\\Accuracy_{n_points}_points_{now}.png",
                dpi=200)
    plt.show()

    # %% md

    #### Create animation to visualize training

    # %%

    def animate_points_and_prediction(itr, act_strategy, dataframe):

        dataframe = dataframe[dataframe["Seed"] == 0]
        df_strategy = dataframe[dataframe["Strategy"] == act_strategy].reset_index()
        df_iteration = df_strategy[df_strategy['Iteration'] == itr]

        a_idx = df_iteration["Active Idx"]
        u_idx = df_iteration["Used Idx"].item()
        new_idx = [1 if idx in a_idx else 0 for idx in u_idx]

        x_0, x_1 = x_t.cpu().numpy()[:, 0], x_t.cpu().numpy()[:, 1]
        preds = df_iteration["Predictions"].item()

        sns.scatterplot(x=x_0, y=x_1, hue=preds, legend=False)
        sns.scatterplot(x=x_0[np.asarray(u_idx)], y=x_1[np.asarray(u_idx)],
                        hue=new_idx, legend=False)
        plt.axhline(0.5, 0, 1, c="k")
        plt.axvline(0.5, 0, 1, c="k")
        plt.title(f"Selected data points by {strategy} strategy, iter {i}")
        plt.xlabel("$x^1$")
        plt.ylabel("$x^2$")


    # %% md

    #### Displaying some pictures from the animations

    # %%

    sns.set(style="ticks", font="Times New Roman", font_scale=1.3,
            rc={'figure.figsize': (6, 5)})
    for strategy in strategies:
        # iterations = [10] if strategy != SUPERVISED else [15]
        iterations = [*range(1, 10)]
        for i in iterations:
            print(f"Iteration {i}/{len(iterations)} {strategy} strategy")

            animate_points_and_prediction(i, strategy, dfs)
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            plt.savefig(f"{image_folder}\\{strategy}_{i}.png")
            plt.show()
