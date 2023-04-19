
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
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.tree import DecisionTreeRegressor

    from kal.knowledge import InsuranceLoss
    from kal.utils import visualize_data_predictions, set_seed

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import tqdm
    from torch.utils.data import TensorDataset
    from kal.active_strategies import SAMPLING_STRATEGIES, DROPOUTS, ADV_DEEPFOOL, ADV_BIM, ENTROPY, \
    ENTROPY_D, BALD, FAST_STRATEGIES, NAME_MAPPINGS_LATEX, NAME_MAPPINGS, REGRESSION_STRATEGIES
    from kal.knowledge.xor import steep_sigmoid
    from kal.network import MLP, train_loop, evaluate, predict_dropout, predict

    plt.rc('animation', html='jshtml')
    plt.close('all')

    dataset_name = "insurance"
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

    #### Loading data for the Insurance dataset

    # %%

    dataset_name = "insurance"
    data_folder = os.path.join("..", "data")
    insurance_dataset = pd.read_csv(os.path.join(data_folder, dataset_name + ".csv"))
    insurance_dataset = pd.get_dummies(insurance_dataset, columns=["sex", "smoker",
                                               "region"])
    insurance_dataset = insurance_dataset.rename(columns={"smoker_yes": "smoker",
                                                          "sex_male": "male",
                                                          "charges": "insurance"})
    X = insurance_dataset.drop(columns=["insurance", "smoker_no", "sex_female"])
    x = MinMaxScaler().fit_transform(X)
    y = insurance_dataset["insurance"].to_numpy()
    y = MinMaxScaler().fit_transform(y.reshape(-1, 1))

    feat_names = X.columns.to_numpy().tolist()
    class_names = ["charges"]

    print("Class names", class_names, "Feat names", feat_names)

    model = DecisionTreeRegressor(max_depth=2, random_state=0)
    # model = DecisionTreeRegressor(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    model = model.fit(x_train, y_train)
    # cross_val_score(model, X, Y, cv=10)

    y_pred = model.predict(x_test)
    print("Tree R2:", r2_score(y_test, y_pred))

    text_representation = tree.export_text(model, feature_names=feat_names)
    print(text_representation)

    #%% md

    #### Visualizing Insurance data

    #%%

    sns.scatterplot(data=insurance_dataset, x="smoker", y="age", hue="insurance")
    plt.savefig(f"{image_folder}\\data_labelling.png")
    plt.show()


    # %%

    x_t = torch.FloatTensor(x)
    y_t = torch.FloatTensor(y)
    y_multi_t = torch.stack((y_t, 1 - y_t), dim=1).squeeze()
    dataset = TensorDataset(x_t, y_t)

    tot_points = x.shape[0]
    input_size = x.shape[1]

    first_points = 10
    n_points = 10
    n_iterations = (300 - first_points) // n_points
    seeds = 10  #
    lr = 1e-3
    epochs = 100
    hidden_size = 100
    load = False

    loss = torch.nn.MSELoss(reduction="none")
    metric = mean_squared_error
    metric = r2_score

    KLoss = partial(InsuranceLoss, names=class_names)
    # strategies = STRATEGIES
    strategies = REGRESSION_STRATEGIES
    # strategies = KALS
    # strategies = DROPOUTS
    print("Strategies:", strategies)
    print("n_points", n_points, "n_iterations", n_iterations)


    # %% md
    #### Computing constraint satisfaction on
    # %%
    k_loss = KLoss()(y_t, x=x_t)
    sns.scatterplot(x=range(y_t.shape[0]), y=y_t[:, 0], hue=k_loss)
    plt.show()

    # %%md
    #### Calculating the prediction of the rule
    # %%

    def calculate_rule_prediction(x: torch.Tensor) -> torch.Tensor:
        smoker = x[:, 4]
        high_bmi = steep_sigmoid(x[:, 1], b=0.4).float()
        old = steep_sigmoid(x[:, 0], b=0.5).float()

        antecedent_low = (1 - smoker) * (1 - old)
        antecedent_mid_low = (1 - smoker) * old
        antecedent_mid_high = smoker * (1 - high_bmi)
        antecedent_high = smoker * high_bmi

        return antecedent_low * 0.1 + antecedent_mid_low * 0.2 + antecedent_mid_high * 0.35 + antecedent_high * 0.65

    pred_rule = calculate_rule_prediction(x_t)
    print("Rule Accuracy:", r2_score(y_t, pred_rule))


    #### Active Learning Strategy Comparison
    dfs = []
    cv = KFold(n_splits=seeds)

    for seed, (train_idx, test_idx) in enumerate(cv.split(x_t, y_t)):
        train_sample = len(train_idx)
        set_seed(seed)
        first_idx = np.random.choice(train_sample, first_points, replace=False).tolist()
        print("First idx", first_idx)

        for strategy in strategies:
            active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss, loss=loss,
                                                            main_classes=[0, 1])
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load:
                df = pd.read_pickle(df_file)
                if "Accuracy" in df.columns:
                    df = df.rename(columns={"Accuracy": "Test Accuracy"}, errors="raise")
                dfs.append(df)
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
                num_classes = 2
                x_train, y_train = x_t[train_idx], y_multi_t[train_idx]
                x_test, y_test = x_t[test_idx], y_multi_t[test_idx]
            else:
                num_classes = 1
                x_train, y_train = x_t[train_idx], y_t[train_idx]
                x_test, y_test = x_t[test_idx], y_t[test_idx]

            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)

            set_seed(0)
            net = MLP(input_size=input_size, hidden_size=hidden_size,
                      n_classes=num_classes, dropout=True, activation=torch.nn.Identity()).to(dev)

            # first training with few randomly selected data
            used_idx = first_idx.copy()
            losses = []
            for it in (pbar := tqdm.trange(1, n_iterations + 1)):

                losses += train_loop(net, train_dataset, used_idx, epochs, device=dev,
                                     lr=lr, loss=loss)

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
                                                   loss=loss)

                t = time.time()
                active_idx, active_loss = active_strategy.selection(preds_t, used_idx, n_points,
                                                                    x=x_t[train_idx], labels=y_t[train_idx],
                                                                    preds_dropout=preds_dropout,
                                                                    clf=net, dataset=train_dataset)
                used_time = (time.time() - t)
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
                                     f"a_l: {active_loss.mean().item():.2f}, "
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
    mean_auc = dfs.groupby("Strategy").mean().round(4)['Test Accuracy']
    std_auc = dfs.groupby(["Strategy", "Seed"]).mean().groupby("Strategy").std().round(4)['Test Accuracy']
    for i, strategy in enumerate(dfs.groupby("Strategy").mean().index):
        print(f"Strategy: {strategy}, AUC: {mean_auc[i].item():.4f}, +-, {std_auc[i].item():.4f}")

    # %%
    #
    # dfs = pd.read_pickle(os.path.join(f"{result_folder}",
    #                                   f"metrics_{n_points}_points_{now}.pkl"))
    dfs['Points'] = [len(used) for used in dfs['Used Idx']]
    ours = ["KAL" in strategy for strategy in dfs['Strategy']]
    dfs['Ours'] = ours

    dfs = dfs.sort_values(['Strategy', 'Seed', 'Iteration'])
    dfs = dfs.reset_index()

    rows = []
    Strategies = []
    for it, row in dfs.iterrows():
        if row['Points'] > (n_points * n_iterations + first_points):
            dfs = dfs.drop(it)
        else:
            Strategies.append(NAME_MAPPINGS_LATEX[row['Strategy']])
    dfs['Strategy'] = Strategies


    sns.set(style="whitegrid", font_scale=1.5,
            rc={'figure.figsize': (10, 8)})
    sns.lineplot(data=dfs, x="Points", y="Test Accuracy",
                 hue="Strategy", style="Ours", size="Ours",
                 legend=False, ci=None, style_order=[1, 0],
                 size_order=[1, 0], sizes=[4,2])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.xlabel("Number of points used")
    # plt.xlim([-10, 400])
    # plt.title("Comparison of the accuracies in the various strategy
    #            in function of the iterations")
    labels = [NAME_MAPPINGS[strategy] for strategy in sorted(strategies)]
    plt.legend(title='Strategy', loc='lower right', labels=labels)
    plt.savefig(f"{image_folder}\\Accuracy_{dataset_name}_{n_points}_points_{now}.png",
                dpi=200)
    plt.show()

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

