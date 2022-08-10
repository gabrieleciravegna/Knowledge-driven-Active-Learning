
if __name__ == "__main__":

    # %% md

    # Knowledge-Driven Active Learning - Experiment on the XOR problem

    # %% md
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import datetime
    import random
    import time
    from functools import partial
    from statistics import mean

    from sklearn import tree
    from sklearn.datasets import load_iris
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    from kal.knowledge import IrisLoss
    from kal.knowledge.knowledge_loss import CombinedLoss
    from kal.utils import visualize_data_predictions, set_seed

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import tqdm
    from torch.utils.data import TensorDataset
    from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, KALS, DROPOUTS, KAL_STAR_DU, KAL_STAR_DROP_DU
    from kal.knowledge.xor import steep_sigmoid
    from kal.metrics import F1
    from kal.network import MLP, train_loop, evaluate, predict_dropout, predict

    plt.rc('animation', html='jshtml')
    plt.close('all')

    dataset_name = "iris"
    model_folder = os.path.join("models", dataset_name)
    result_folder = os.path.join("results", dataset_name)
    image_folder = os.path.join("images", dataset_name)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    if not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    sns.set_theme(style="whitegrid", font="Times New Roman")
    now = str(datetime.datetime.now()).replace(":", ".")
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Working on {dev}")

    # %% md

    #### Loading data for the IRIS dataset

    # %%

    iris_dataset = load_iris()
    X = iris_dataset.data
    Y = iris_dataset.target
    feat_names = iris_dataset.feature_names
    class_names = iris_dataset.target_names
    print("Class names", class_names, "Feat names", feat_names)

    x = MinMaxScaler().fit_transform(X)
    y = OneHotEncoder(sparse=False).fit_transform(Y.reshape(-1, 1))
    clf = tree.DecisionTreeClassifier(max_depth=2, random_state=1234)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)
    model = clf.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Tree Accuracy:", f1_score(y_test, y_pred, average="macro") * 100)
    text_representation = tree.export_text(model, feature_names=feat_names)
    print(text_representation)

    # %%

    x_t = torch.FloatTensor(x)
    y_t = torch.FloatTensor(y)
    dataset = TensorDataset(x_t, y_t)

    tot_points = x.shape[0]
    input_size = x.shape[1]
    n_classes = y.shape[1]

    first_points = 5
    n_points = 5
    n_iterations = (75 - first_points) // n_points
    seeds = 10  #
    lr = 3 * 1e-3
    epochs = 200
    hidden_size = 100
    load = True

    KLoss = partial(IrisLoss, names=class_names)
    strategies = STRATEGIES
    # strategies = KALS
    # strategies = DROPOUTS
    print("Strategies:", strategies)
    print("n_points", n_points, "n_iterations", n_iterations)

    #%% md

    #### Visualizing iris data

    #%%

    sns.scatterplot(x=x[:, 2], y=x[:, 3], hue=Y)
    plt.savefig(f"{image_folder}\\data_labelling.png")
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
    sns.scatterplot(x=x_t[:, 2].numpy(), y=x_t[:, 3].numpy(), hue=pred_rule.argmax(dim=1),
                    style=y_t.argmax(dim=1) == pred_rule.argmax(dim=1))
    plt.show()


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
                                                            main_classes=[0, 1, 2])
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

            x_train, y_train = x_t[train_idx], y_t[train_idx]
            x_test, y_test = x_t[test_idx], y_t[test_idx]
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
            if strategy in [KAL_STAR_DU, KAL_STAR_DROP_DU]:
                loss = CombinedLoss(KLoss, lambda_val=0.5)
            else:
                loss = torch.nn.BCEWithLogitsLoss(reduction="none")
            metric = F1()

            set_seed(0)
            net = MLP(input_size=input_size, hidden_size=hidden_size,
                      n_classes=n_classes, dropout=True).to(dev)

            # first training with few randomly selected data
            used_idx = first_idx.copy()
            losses = []
            for it in (pbar := tqdm.trange(1, n_iterations + 1)):
                t = time.time()

                losses += train_loop(net, train_dataset, used_idx, epochs,
                                     lr=lr, loss=loss)

                preds_t = predict(net, train_dataset)
                if strategy in DROPOUTS:
                    preds_dropout = predict_dropout(net, train_dataset)
                    assert (preds_dropout - preds_t).abs().sum() > .0, \
                        "Error in computing dropout predictions"
                else:
                    preds_dropout = None

                test_accuracy, sup_loss = evaluate(net, test_dataset, metric=metric,
                                                   loss=torch.nn.BCEWithLogitsLoss(reduction="none"))

                active_idx, active_loss = active_strategy.selection(preds_t, used_idx, n_points,
                                                                    x=x_t[train_idx], labels=y_t[train_idx],
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
    #
    # dfs = pd.read_pickle(os.path.join(f"{result_folder}",
    #                                   f"metrics_{n_points}_points_{now}.pkl"))
    # dfs['Points'] = [len(used) for used in dfs['Used Idx']]
    # ours = ["KAL" in strategy for strategy in dfs['Strategy']]
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
    #
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
    #
    # # %%
    #
    # sns.set(style="whitegrid", font_scale=1.5,
    #         rc={'figure.figsize': (10, 8)})
    # sns.lineplot(data=dfs, x="Points", y="Accuracy",
    #              hue="Strategy", style="Ours", size="Ours",
    #              legend=False, ci=None, style_order=[1, 0],
    #              size_order=[1, 0], sizes=[4,2])
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

    #### Displaying some pictures to visualize training

    # %%
    dfs = pd.read_pickle(f"{result_folder}\\results.pkl")

    sns.set(style="ticks", font="Times New Roman", font_scale=1.3,
            rc={'figure.figsize': (6, 5)})
    for strategy in strategies:
        # iterations = [10] if strategy != SUPERVISED else [15]
        iterations = [*range(1, 10)]
        for i in iterations:
            print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
            png_file = os.path.join(f"{image_folder}", f"{strategy}_{i}.png")
            # if not os.path.exists(png_file):
            visualize_data_predictions(x_t, i, strategy, dfs, png_file,
                                       dimensions=[2, 3], dataset="iris")

