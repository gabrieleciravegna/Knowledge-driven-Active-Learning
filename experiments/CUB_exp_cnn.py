
if __name__ == "__main__":

    #%% md

    # Constrained Active Learning - Experiment on the CUB200 problem

    #%% md

    #### Importing libraries


    #%%

    #%matplotlib inline
    #%autosave 10

    import os

    gpu_id = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    import random
    import datetime
    import time
    from statistics import mean

    from kal.utils import visualize_active_vs_sup_loss, set_seed

    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from torchvision.transforms import transforms
    import tqdm
    from functools import partial

    from sklearn.model_selection import StratifiedKFold

    from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, ADV_DEEPFOOL, ADV_BIM, ENTROPY, ENTROPY_D, BALD, \
    RANDOM, KAL_U, SUPERVISED, UNCERTAINTY, DROPOUTS
    from kal.knowledge import CUB200Loss
    from kal.network import MLP, train_loop, evaluate, predict, predict_dropout, ResNet10

    from data.Cub200 import CUBDataset
    from data.cub200 import classes
    from kal.metrics import F1

    plt.rc('animation', html='jshtml')

    dataset = "cub200"
    model_folder = os.path.join("models", dataset)
    result_folder = os.path.join("results", dataset)
    image_folder = os.path.join("images", dataset)
    data_folder = os.path.join("../data", dataset)

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

    #%% md

    #### Loading data for the cub200 problem.
    # Data pass through a RESNET 18 first which extract the data features

    #%%

    first_points = 2000
    n_points = 200
    n_iterations = 10
    seeds = 5
    epochs = 20
    num_classes = 308
    hidden_size = num_classes * 2
    lr = 1e-3
    batch_size = 64
    metric = F1()
    load = False

    # strategies = STRATEGIES
    strategies = [RANDOM, KAL_U, SUPERVISED, UNCERTAINTY]
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
    dataset = CUBDataset(data_folder, transform)
    main_classes = dataset.main_classes

    # %%
    #### Visualizing and checking knowledge loss on the labels

    KLoss = partial(CUB200Loss, main_classes=dataset.main_classes,
                    attributes=dataset.attributes,
                    combinations=dataset.class_attr_comb
        )
    y_t = torch.as_tensor(dataset.targets)
    cons_loss = KLoss()(y_t).sort()[0].cpu().numpy()
    sns.scatterplot(x=[*range(len(cons_loss))], y=cons_loss)
    plt.show()

    # %%
    #### Active Learning Strategy Comparison
    dfs = []
    skf = StratifiedKFold(n_splits=seeds)

    for seed, (train_idx, test_idx) in enumerate(skf.split(y_t, y_t.argmax(dim=1).cpu())):
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
                "Test Idx": [],
            }

            if strategy in DROPOUTS:
                dropout = True
            else:
                dropout = False

            train_dataset = Subset(dataset, train_idx)
            test_dataset = Subset(dataset, test_idx)
            loss = torch.nn.BCEWithLogitsLoss(reduction="none")
            metric = F1()

            set_seed(0)
            net = ResNet10(n_classes=num_classes).to(dev)

            # first training with few randomly selected data
            used_idx = first_idx.copy()
            losses = train_loop(net, train_dataset, used_idx, epochs, batch_size,
                                lr=lr, loss=loss, visualize_loss=False, device=dev, verbose=True)
            test_accuracy, sup_loss = evaluate(net, test_dataset, batch_size,
                                               metric=metric, loss=loss, device=dev)

            for it in (pbar := tqdm.trange(1, n_iterations + 1)):
                pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                     f"acc: {np.mean([0] + df['Accuracy']):.2f}, s_l: {mean(sup_loss):.2f}, "
                                     f"l: {losses[-1]:.2f}, p: {len(used_idx)}")
                t = time.time()

                preds_t = predict(net, train_dataset, batch_size, device=dev)
                preds_dropout = predict_dropout(net, train_dataset, batch_size, device=dev)

                assert not dropout or (preds_dropout - preds_t).abs().sum() > 1.0, \
                    "Error in computing dropout predictions"

                test_accuracy, sup_loss = evaluate(net, test_dataset, batch_size,
                                                   metric=metric, loss=loss, device=dev)

                active_idx, active_loss = active_strategy.selection(preds_t, used_idx, n_points,
                                                                    labels=y_t[train_idx],
                                                                    preds_dropout=preds_dropout,
                                                                    clf=net, dataset=train_dataset,
                                                                    main_classes=main_classes,
                                                                    device=dev)
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
                                         batch_size, lr=lr, loss=loss, device=dev)
                else:
                    pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                         f"{np.mean([0] + df['Accuracy']):.2f}, "
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
                 hue="Strategy", ci=None)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.ylabel("Accuracy")
    plt.xlabel("Number of points used")
    # plt.xlim([-10, 400])
    # plt.title("Comparison of the accuracies in the various strategy
    #            in function of the iterations")
    plt.savefig(f"{image_folder}\\Accuracy_{n_points}_points_{now}.png",
                dpi=200)
    plt.show()

    # %% md

    #### Displaying some pictures from the animations

    # %%
    #
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
