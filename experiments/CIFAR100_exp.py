from data.Cifar100 import CLASS_1_HOTS
from kal.knowledge.cifar100 import CIFAR100Loss

if __name__ == "__main__":

    #%% md

    # Constrained Active Learning - Experiment on the CUB200 problem

    #%% md

    #### Importing libraries

    #%%

    #%matplotlib inline
    #%autosave 10

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
    KALS, MARGIN, MARGIN_D, DROPOUTS, KAL_PLUS_DROP_DU, FAST_STRATEGIES, KMEANS, KCENTER, TO_RERUN
    from kal.network import MLP, train_loop, evaluate, predict, predict_dropout
    from kal.utils import visualize_active_vs_sup_loss, set_seed

    from data.Cub200 import CUBDataset
    from data.Cifar100 import classes
    from kal.metrics import F1
    from kal.knowledge import CUB200Loss

    plt.rc('animation', html='jshtml')

    dataset = "cifar100"
    model_folder = os.path.join("models", dataset)
    result_folder = os.path.join("results", dataset)
    image_folder = os.path.join("images", dataset)
    data_folder = os.path.join("..", "data", dataset)
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
    KLoss = partial(CIFAR100Loss, names=classes)

    #%% md

    #### Loading data for the cub200 problem.
    # Data pass through a RESNET 18 first which extract the data features

    #%%

    first_points = 1000
    n_points = 100
    n_iterations = 90
    seeds = 5
    epochs = 100
    num_classes = 120
    hidden_size = num_classes * 2
    lr = 1e-3
    metric = F1()
    load = True

    # strategies = [KMEANS]
    strategies = STRATEGIES[::-1][2:]
    strategies.pop(strategies.index(KCENTER))
    strategies.pop(strategies.index(KMEANS))
    # strategies = FAST_STRATEGIES
    # strategies = [ENTROPY_D, ENTROPY, MARGIN_D, MARGIN, ]
    # strategies = KALS[::-1]
    # strategies = [KAL_PLUS_DROP_DU]
    print("Strategies:", strategies)
    print("n_points", n_points, "n_iterations", n_iterations)

    #%% md
    #### Loading data for the cub200 problem.
    # Data pass through a RESNET 50 first which extract the data features
    #%%
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    annoying_dir = os.path.join(data_folder, "__pycache__")
    if os.path.isdir(annoying_dir):
        shutil.rmtree(annoying_dir)
    dataset = torchvision.datasets.CIFAR100(data_folder,
                                            download=True,
                                            transform=transform,
                                            train=True)
    dataset.main_classes = [*range(100)]
    feature_extractor = torchvision.models.resnet50(pretrained=True)
    feature_extractor.fc = torch.nn.Identity()
    data_loader = DataLoader(dataset, batch_size=128, num_workers=8)
    tot_points = len(dataset)

    #%%

    feature_file = os.path.join(data_folder, "ResNet50-TL-feats.pth")
    if os.path.isfile(feature_file):
        x = torch.load(feature_file)
        y = [CLASS_1_HOTS[dataset.classes[t]] for t in dataset.targets]
        print("Features loaded")
    else:
        x, y = [], []
        with torch.no_grad():
            feature_extractor.eval(), feature_extractor.to(dev)
            for batch_data, batch_labels in tqdm.tqdm(data_loader):
                batch_x = feature_extractor(batch_data.to(dev))
                x.append(batch_x)
                y.append(batch_labels)
            x = torch.cat(x)
            torch.save(x, feature_file)
            y = [CLASS_1_HOTS[dataset.classes[t]] for t in dataset.targets]
    input_size = x.shape[1]


    #%%
    #### Visualizing and checking knowledge loss on the labels

    x_t = torch.as_tensor(x, dtype=torch.float).to(dev)
    y_t = torch.as_tensor(y, dtype=torch.float).to(dev)
    cons_loss = KLoss()(y_t, targets=True).sort()[0].cpu().numpy()
    sns.scatterplot(x=[*range(len(cons_loss))], y=cons_loss)
    plt.show()

    #%%
    #### Active Learning Strategy Comparison
    dfs = []
    skf = StratifiedKFold(n_splits=seeds)

    for strategy in strategies:
        for seed, (train_idx, test_idx) in enumerate(skf.split(x_t.cpu(), y_t.argmax(dim=1).cpu())):
            train_sample = len(train_idx)
            set_seed(seed)
            first_idx = np.random.choice(train_sample, first_points, replace=False).tolist()
            # print("First idx", first_idx)

            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load and strategy not in TO_RERUN:
                df = pd.read_pickle(df_file)
                if "Predictions" in df:
                    df.pop("Predictions")
                    df.to_pickle(df_file)
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
                # "Predictions": [],
                "Train Accuracy": [],
                "Test Accuracy": [],
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

            active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss,
                                                            main_classes=dataset.main_classes)

            net = MLP(input_size=input_size, hidden_size=hidden_size,
                      n_classes=num_classes, dropout=True).to(dev)

            # first training with few randomly selected data
            losses = []
            used_idx = first_idx.copy()
            for it in (pbar := trange(0, n_iterations)):
                t = time.time()

                losses += train_loop(net, train_dataset, used_idx, epochs * 2 if it == 0 else epochs,
                                     lr=lr, loss=loss, device=dev)
                train_accuracy, _, preds_t = evaluate(net, train_dataset, loss=loss,
                                                      device=dev, return_preds=True)
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
                                                                    main_classes=dataset.main_classes)
                used_idx += active_idx

                df["Strategy"].append(strategy)
                df["Seed"].append(seed)
                df["Iteration"].append(it)
                df["Active Idx"].append(active_idx.copy())
                df["Used Idx"].append(used_idx.copy())
                # df["Predictions"].append(preds_t.cpu().numpy())
                df["Train Accuracy"].append(train_accuracy)
                df["Test Accuracy"].append(test_accuracy)
                df["Supervision Loss"].append(sup_loss)
                df["Active Loss"].append(active_loss.cpu().numpy())
                df["Time"].append((time.time() - t))
                df["Train Idx"].append(train_idx)
                df["Test Idx"].append(test_idx)

                assert isinstance(used_idx, list), "Error"

                pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                     f"train acc: {np.mean(df['Train Accuracy']):.2f}, "
                                     f"test acc: {np.mean(df['Test Accuracy']):.2f}, "
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

    dfs_auc = dfs.groupby("Strategy").mean()['Test Accuracy']
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
    sns.lineplot(data=dfs, x="Points", y='Test Accuracy',
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

    #### Displaying some pictures from the animations

    # %%

    sns.set(style="ticks", font="Times New Roman", font_scale=1.3,
            rc={'figure.figsize': (6, 5)})
    for strategy in strategies:
        # iterations = [10] if strategy != SUPERVISED else [15]
        iterations = [*range(1, 10)]
        for i in iterations:
            print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
            png_file = os.path.join(f"{image_folder}", f"{strategy}_{i}.png")
            # if not os.path.exists(png_file):
            visualize_active_vs_sup_loss(x_t, i, strategy, dfs, png_file, )

