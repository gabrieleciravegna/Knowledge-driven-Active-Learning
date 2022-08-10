
if __name__ == "__main__":

    #%% md

    # Constrained Active Learning - Experiment on the ANIMALS problem

    #%% md

    #### Importing libraries

    #%%

    #%matplotlib inline
    #%autosave 10

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTHONPYCACHEPREFIX"] = os.path.join("..", "__pycache__")

    import tqdm
    import shutil
    import random
    import datetime
    import time
    from functools import partial
    from statistics import mean

    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.transforms import transforms
    from tqdm import trange
    from sklearn.model_selection import StratifiedKFold

    from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, ADV_DEEPFOOL, ADV_BIM, ENTROPY, ENTROPY_D, BALD, \
    KALS, MARGIN, MARGIN_D, DROPOUTS, KAL_PLUS_DROP_DU, KCENTER, KMEANS, NAME_MAPPINGS_LATEX, RANDOM, NAME_MAPPINGS, \
    KAL_PLUS_DU, KAL_PLUS
    from kal.network import MLP, train_loop, evaluate, predict, predict_dropout
    from kal.utils import visualize_active_vs_sup_loss, set_seed

    from data.Animals import CLASS_1_HOTS, classes
    from kal.metrics import F1
    from kal.knowledge import AnimalLoss

    plt.rc('animation', html='jshtml')

    dataset_name = "animals"
    model_folder = os.path.join("models", dataset_name)
    result_folder = os.path.join("results", dataset_name)
    image_folder = os.path.join("images", dataset_name)
    data_folder = os.path.join("..", "data", "Animals")
    assert os.path.isdir(data_folder), "Data not available in the required folder"

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
    KLoss = partial(AnimalLoss, names=classes)

    #%%
    first_points = 100
    n_points = 50
    n_iterations = 48
    seeds = 5
    hidden_size = 100
    lr = 1e-3
    epochs = 250
    main_classes = range(7)
    metric = F1()
    load = True

    # strategies = [BALD]
    strategies = STRATEGIES
    # # strategies.pop(strategies.index(KMEANS))
    # # strategies.pop(strategies.index(KCENTER))
    strategies.pop(strategies.index(ADV_DEEPFOOL))
    # strategies.pop(strategies.index(ADV_BIM))
    # strategies = [ENTROPY_D, ENTROPY, MARGIN_D, MARGIN, ]
    # strategies = KALS[::-1]
    strategies = KALS
    print("Strategies:", strategies)
    print("n_points", n_points, "n_iterations", n_iterations)

    #%% md

    #### Loading data for the animal's problem.
    # Data pass through a RESNET 50 first which extract the data features

    #%%
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    annoying_dir = os.path.join(data_folder, "__pycache__")
    if os.path.isdir(annoying_dir):
        shutil.rmtree(annoying_dir)
    dataset = torchvision.datasets.ImageFolder(data_folder, transform=transform)

    feature_extractor = torchvision.models.resnet50(pretrained=True)
    feature_extractor.fc = torch.nn.Identity()
    data_loader = DataLoader(dataset, batch_size=128, num_workers=8)
    tot_points = len(dataset)
    class_names = classes
    n_classes = len(class_names)

    feature_file = os.path.join(data_folder, "ResNet50-TL-feats.pth")
    if os.path.isfile(feature_file):
        x = torch.load(feature_file)
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

    #%%
    #### Visualizing and checking knowledge loss on the labels

    x_t = torch.as_tensor(x, dtype=torch.float).to(dev)
    y_t = torch.as_tensor(y_multi, dtype=torch.float).to(dev)
    cons_loss = KLoss()(y_t).sort()[0].cpu().numpy()
    sns.scatterplot(x=[*range(len(cons_loss))], y=cons_loss)
    plt.show()

    #%%
    #### Active Learning Strategy Comparison
    dfs = []
    skf = StratifiedKFold(n_splits=seeds)

    for seed, (train_idx, test_idx) in enumerate(skf.split(x_t.cpu(), y_t.argmax(dim=1).cpu())):
        train_sample = len(train_idx)
        set_seed(seed)
        first_idx = np.random.choice(train_sample, first_points, replace=False).tolist()
        print("First idx", first_idx)

        for strategy in strategies:
            active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss,
                                                            main_classes=main_classes)
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load and KAL_PLUS not in strategy:
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
                                     lr=lr, loss=loss, device=dev)

                preds_t = predict(net, train_dataset, device=dev)
                if strategy in DROPOUTS:
                    preds_dropout = predict_dropout(net, train_dataset, device=dev)
                    assert (preds_dropout - preds_t).abs().sum() > .0, \
                        "Error in computing dropout predictions"
                else:
                    preds_dropout = None

                test_accuracy, sup_loss = evaluate(net, test_dataset, metric=metric, loss=loss, device=dev)

                active_idx, active_loss = active_strategy.selection(preds_t, used_idx, n_points,
                                                                    x=x_t[train_idx], labels=y_t[train_idx],
                                                                    preds_dropout=preds_dropout,
                                                                    clf=net, dataset=train_dataset,
                                                                    main_classes=main_classes)
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

    dfs = pd.read_pickle(os.path.join(f"{result_folder}",
                                      f"metrics_{n_points}_points_{now}.pkl"))
    dfs['Points'] = [len(used) for used in dfs['Used Idx']]
    ours = ["KAL" in strategy for strategy in dfs['Strategy']]
    dfs['Ours'] = ours

    dfs = dfs.sort_values(['Strategy', 'Seed', 'Iteration'])
    dfs = dfs.reset_index()

    rows = []
    Strategies = []
    for i, row in dfs.iterrows():
        if row['Points'] > (n_points * n_iterations + first_points):
            dfs = dfs.drop(i)
        else:
            Strategies.append(NAME_MAPPINGS_LATEX[row['Strategy']])
    dfs['Strategy'] = Strategies

    aucs = []
    dfs_auc_mean = dfs.groupby("Strategy").mean()['Accuracy'].tolist()
    dfs_auc_std = dfs.groupby(["Strategy", "Seed"]).mean()['Accuracy'] \
        .groupby("Strategy").std().tolist()
    for mean, std in zip(dfs_auc_mean, dfs_auc_std):
        auc = f"${mean:.2f}$ {{\\tiny $\\pm {std:.2f}$ }}"
        aucs.append(auc)
    df_auc = pd.DataFrame({
        "Strategy": np.unique(Strategies),
        "AUC": aucs,
    }).set_index("Strategy")
    print(df_auc.to_latex(float_format="%.2f", escape=False))
    with open(os.path.join(f"{result_folder}",
                           f"table_auc_latex_{now}.txt"), "w") as f:
        f.write(df_auc.to_latex(float_format="%.2f", escape=False))

    final_accs = []
    dfs_final_accs = dfs[dfs['Iteration'] == n_iterations - 1]
    final_accs_mean = dfs_final_accs.groupby("Strategy").mean()['Accuracy'].tolist()
    final_accs_std = dfs_final_accs.groupby("Strategy").std()['Accuracy'].tolist()
    for mean, std in zip(final_accs_mean, final_accs_std):
        final_acc = f"${mean:.2f}$ {{\\tiny $\\pm {std:.2f}$ }}"
        final_accs.append(final_acc)
    df_final_acc = pd.DataFrame({
        "Strategy": np.unique(Strategies),
        "Final F1": final_accs,
    }).set_index("Strategy")
    print(df_final_acc.to_latex(float_format="%.2f", escape=False))
    with open(os.path.join(f"{result_folder}",
                           f"table_final_acc_latex_{now}.txt"), "w") as f:
        f.write(df_final_acc.to_latex(float_format="%.2f", escape=False))

    times = []
    dfs_time_mean = dfs.groupby("Strategy").mean()['Time']
    base_time = dfs_time_mean[dfs_time_mean.index == RANDOM].item()
    for time in dfs_time_mean.tolist():
        time = time / base_time
        time = time if time >= 1. else 1.
        time = f"${time:.2f}$ x"
        times.append(time)
    df_times = pd.DataFrame({
        "Strategy": np.unique(Strategies),
        "Times": times
    }).set_index("Strategy")
    print(df_times.to_latex(float_format="%.2f", escape=False))
    with open(os.path.join(f"{result_folder}",
                           f"table_times_latex_{now}.txt"), "w") as f:
        f.write(df_times.to_latex(float_format="%.2f", escape=False))

    # %%

    sns.set(style="whitegrid", font_scale=1.5,
            rc={'figure.figsize': (10, 8)})
    sns.lineplot(data=dfs, x="Points", y="Accuracy",
                 hue="Strategy", style="Ours", size="Ours",
                 legend=False, ci=None, style_order=[1, 0],
                 size_order=[1, 0], sizes=[4,2])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.ylabel("Accuracy")
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

    # sns.set(style="ticks", font="Times New Roman", font_scale=1.3,
    #         rc={'figure.figsize': (6, 5)})
    # for strategy in strategies:
    #     # iterations = [10] if strategy != SUPERVISED else [15]
    #     iterations = [*range(1, 10)]
    #     for i in iterations:
    #         print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
    #         png_file = os.path.join(f"{image_folder}", f"{strategy}_{i}.png")
    #         # if not os.path.exists(png_file):
    #         visualize_active_vs_sup_loss(x_t, i, strategy, dfs, png_file, )
    #
