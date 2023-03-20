
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

    from kal.active_strategies import SAMPLING_STRATEGIES, ADV_DEEPFOOL, KALS, DROPOUTS, KAL_XAI_DU, KAL_XAI_DROP_DU
    from kal.network import MLP, train_loop, evaluate, predict, predict_dropout
    from kal.utils import set_seed

    from data.Animals import CLASS_1_HOTS, classes
    from kal.metrics import F1
    from kal.knowledge import AnimalLoss

    plt.rc('animation', html='jshtml')

    dataset_name = "animals"
    model_folder = os.path.join("models", dataset_name)
    result_folder = os.path.join("results", dataset_name)
    image_folder = os.path.join("images", dataset_name)
    data_folder = os.path.join("..", "data", dataset_name)
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

    #%%
    first_points = 100
    n_points = 50
    rand_points = 20
    n_iterations = 48
    seeds = 5
    hidden_size = 100
    lr = 1e-3
    epochs = 250
    main_classes = range(7)
    discretize_feats = False
    metric = F1()
    load = False
    print("Rand points", rand_points)

    # strategies = [BALD]
    strategies = [KAL_XAI_DU]
    # # strategies.pop(strategies.index(KMEANS))
    # # strategies.pop(strategies.index(KCENTER))
    # strategies.pop(strategies.index(ADV_DEEPFOOL))
    # strategies.pop(strategies.index(ADV_BIM))
    # strategies = [ENTROPY_D, ENTROPY, MARGIN_D, MARGIN, ]
    # strategies = KALS[::-1]
    # strategies = KALS
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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

    #%%
    #### Visualizing and checking knowledge loss on the labels
    KLoss = partial(AnimalLoss, names=classes)

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
                                                            main_classes=main_classes,
                                                            rand_points=rand_points,
                                                            hidden_size=hidden_size,
                                                            dev=dev, cv=True,
                                                            discretize_feats=discretize_feats,
                                                            class_names=class_names)
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load:
                df = pd.read_pickle(df_file)
                dfs.append(df)
                auc = df['Test Accuracy'].mean()
                print(f"Already trained {df_file}, auc: {auc:.2f}")
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

            # Creating dataset for training and testing
            x_train, y_train = x_t[train_idx], y_t[train_idx]
            x_test, y_test = x_t[test_idx], y_t[test_idx]
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)

            loss = torch.nn.BCEWithLogitsLoss(reduction="none")
            metric = F1()

            set_seed(0)
            net = MLP(n_classes, input_size, hidden_size, dropout=True).to(dev)

            # first training with few randomly selected data
            losses = []
            used_idx = first_idx.copy()
            for it in (pbar := tqdm.trange(1, n_iterations + 1)):
                t = time.time()

                losses += train_loop(net, train_dataset, used_idx, epochs,
                                     lr=lr, loss=loss, device=dev)
                train_accuracy, _, preds_t = evaluate(net, train_dataset, loss=loss, device=dev,
                                                      labelled_idx=used_idx, return_preds=True)
                if strategy in DROPOUTS:
                    preds_dropout = predict_dropout(net, train_dataset, device=dev)
                    assert (preds_dropout - preds_t).abs().sum() > .0, \
                        "Error in computing dropout predictions"
                else:
                    preds_dropout = None

                test_accuracy, sup_loss = evaluate(net, test_dataset, metric=metric, loss=loss, device=dev)

                active_idx, active_loss = active_strategy.selection(preds_t, used_idx,
                                                                    n_points, labels=y_train,
                                                                    preds_dropout=preds_dropout,
                                                                    clf=net, dataset=train_dataset,
                                                                    )

                used_idx += active_idx

                df["Strategy"].append(strategy)
                df["Seed"].append(seed)
                df["Iteration"].append(it)
                df["Active Idx"].append(active_idx.copy())
                df["Used Idx"].append(used_idx.copy())
                df["Predictions"].append(preds_t.cpu().numpy())
                df["Train Accuracy"].append(train_accuracy)
                df["Test Accuracy"].append(test_accuracy)
                df["Supervision Loss"].append(sup_loss)
                df["Active Loss"].append(active_loss.cpu().numpy())
                df["Time"].append((time.time() - t))
                df["Train Idx"].append(train_idx)
                df["Test Idx"].append(test_idx)

                assert isinstance(used_idx, list), "Error"

                pbar.set_description(f"{strategy} {seed + 1}/{seeds}, "
                                     f"train auc: {np.mean(df['Train Accuracy']):.2f}, "
                                     f"test auc: {np.mean(df['Test Accuracy']):.2f}, "
                                     # f"s_l: {mean(sup_loss):.2f}, "
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

    mean_auc = dfs.groupby("Strategy").mean().round(2)['Test Accuracy']
    std_auc = dfs.groupby(["Strategy", "Seed"]).mean().groupby("Strategy").std().round(2)['Test Accuracy']
    print("AUC", mean_auc, "+-", std_auc)
    # %% md

    #### Displaying some pictures to visualize training

    # %%

    # sns.set(style="ticks", font="Times New Roman", font_scale=1.3,
    #         rc={'figure.figsize': (6, 5)})
    # for strategy in strategies:
    #     # iterations = [10] if strategy != SUPERVISED else [15]
    #     iterations = [*range(1, 10)]
    #     for it in iterations:
    #         print(f"Iteration {i}/{len(iterations)} {strategy} strategy")
    #         png_file = os.path.join(f"{image_folder}", f"{strategy}_{i}.png")
    #         # if not os.path.exists(png_file):
    #         visualize_active_vs_sup_loss(x_t, it, strategy, dfs, png_file, )
    #
