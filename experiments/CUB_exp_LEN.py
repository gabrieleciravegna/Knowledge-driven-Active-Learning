from kal.knowledge.expl_to_loss import Expl_2_Loss_CV
from kal.losses import EntropyLoss

if __name__ == "__main__":

    # %% md

    # Constrained Active Learning - Experiment on the CUB200 problem

    # %% md

    #### Importing libraries

    # %%

    # %matplotlib inline
    # %autosave 10

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
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from torchvision.transforms import transforms
    from tqdm import trange
    from sklearn.model_selection import StratifiedKFold

    from kal.active_strategies import STRATEGIES, SAMPLING_STRATEGIES, KALS, DROPOUTS, RandomSampling, KAL_LEN_DU, \
    KAL_PARTIAL, KAL_LENS
    from kal.network import MLP, train_loop, evaluate, predict_dropout, ELEN
    from kal.utils import visualize_active_vs_sup_loss, set_seed

    from data.Cub200 import CUBDataset
    from data.CUB200 import classes
    from kal.metrics import F1
    from kal.knowledge import CUB200Loss

    plt.rc('animation', html='jshtml')

    dataset_name = "CUB200"
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
    KLoss = partial(CUB200Loss, names=classes)

    # %% md

    #### Loading data for the cub200 problem.
    # Data pass through a RESNET 18 first which extract the data features

    # %%

    first_points = 2000
    n_points = 200
    n_iterations = 25
    seeds = 5
    rand_points = 80
    epochs = 100
    num_classes = 308
    main_classes = range(200)
    hidden_size = num_classes * 2
    lr = 1e-3
    metric = F1()
    load = True
    mutual_excl = True

    # strategies = KAL_LENS
    # strategies = KAL_PARTIAL
    strategies = STRATEGIES[:-1]
    strategies += [s for s in KAL_PARTIAL if s not in STRATEGIES]
    # strategies =
    # strategies = KALS[::-1]
    # strategies = [KAL_LEN_DU]
    # strategies = FAST_STRATEGIES
    # strategies = [ENTROPY_D, ENTROPY, MARGIN_D, MARGIN, ]
    # strategies = KALS[::-1]
    # strategies = [KAL_PLUS_DROP_DU]
    print("Strategies:", strategies)
    print("n_points", n_points, "n_iterations", n_iterations)

    # %% md
    #### Loading data for the cub200 problem.
    # Data pass through a RESNET 50 first which extract the data features
    # %%
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
    dataset = CUBDataset(data_folder, transform)

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
            torch.save(x, feature_file)
    input_size = x.shape[1]

    # %%
    #### Visualizing and checking knowledge loss on the labels

    x_t = torch.as_tensor(x, dtype=torch.float).to(dev)
    y_t = torch.as_tensor(y, dtype=torch.float).to(dev)
    cons_loss = KLoss(main_classes=dataset.main_classes,
                      attributes=dataset.attributes,
                      combinations=dataset.class_attr_comb)(y_t).sort()[0].cpu().numpy()
    sns.scatterplot(x=[*range(len(cons_loss))], y=cons_loss)
    plt.show()

    # %%
    #### Active Learning Strategy Comparison
    dfs = []
    skf = StratifiedKFold(n_splits=seeds)

    for seed, (train_idx, test_idx) in enumerate(skf.split(x_t.cpu(), y_t.argmax(dim=1).cpu())):
        train_sample = len(train_idx)
        set_seed(seed)
        first_idx = np.random.choice(train_sample, first_points, replace=False).tolist()
        print("First idx", first_idx)

        for strategy in strategies:
            df_file = os.path.join(result_folder, f"metrics_{n_points}_points_"
                                                  f"{seed}_seed_{strategy}_strategy.pkl")
            if os.path.exists(df_file) and load:
                try:
                    df = pd.read_pickle(df_file)
                    if "Predictions" in df.columns:
                        df['Iteration'] = [i for i in range(1, n_iterations + 1)]
                        dfs.append(df)
                        auc = df['Test Accuracy'].mean()
                        print(f"Already trained {df_file}, auc: {auc:.2f}")
                        continue
                except EOFError:
                    print(f"Error in loading file {df_file}, recomputing model")

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

            # Creating datasets and models for the explanations
            # The dataset has labels the explained classes and the attributes as features
            expl_feats = y_train[:, len(main_classes):]
            expl_label = y_train[:, :len(main_classes)]  # [y_train[:, it].unsqueeze(dim=1) for it in range(n_classes)]
            expl_names = class_names[len(main_classes):]
            expl_dataset = TensorDataset(expl_feats, expl_label)  # for it in range(n_classes)

            # first training with few randomly selected data
            losses = []
            used_idx = first_idx.copy()
            for it in (pbar := tqdm.trange(1, n_iterations + 1)):
                t = time.time()

                losses += train_loop(net, train_dataset, used_idx, epochs,
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

                if "LEN" in strategy:
                    expl_model = ELEN(input_size=n_classes - len(main_classes),
                                      hidden_size=len(main_classes) // 10, n_classes=len(main_classes),
                                      dropout=True, activation=torch.nn.Softmax(dim=1)).to(dev)
                    e_loss = EntropyLoss(expl_model, sup_loss=torch.nn.CrossEntropyLoss())
                    train_loop(expl_model, expl_dataset, used_idx, epochs * 5,
                               lr=lr, loss=e_loss, visualize_loss=True, device=dev)
                    explainer_acc, _, _ = evaluate(expl_model, Subset(expl_dataset, used_idx), loss=loss,
                                                   device=dev, return_preds=True)
                    print(f"Explainer acc: {explainer_acc}")
                    formulas, expl_accs = [], []
                    for i in main_classes:
                        formula, expl_acc = expl_model.explain(expl_feats, expl_label, used_idx,
                                                               expl_names, target_class=i,
                                                               return_acc=True)
                        # print(f"{it}) {class_names[it]} <-> {formula}, acc: {expl_acc * 100} %")
                        formulas.append(formula), expl_accs.append(expl_acc)
                    assert mean(expl_accs) * 100 > 90, f"Error in extracting formulas, expl_accs: {expl_accs * 100}"
                    KLoss = partial(Expl_2_Loss_CV, class_names, formulas,
                                    main_classes=main_classes, mutual_excl=mutual_excl)
                else:
                    if "0" in strategy or "25" in strategy or "50" in strategy or "75" in strategy:
                        percentage = int(strategy[-2:])
                    else:
                        percentage = None
                    KLoss = partial(CUB200Loss, main_classes=dataset.main_classes,
                                    attributes=dataset.attributes,
                                    combinations=dataset.class_attr_comb,
                                    percentage=percentage
                                    )

                active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss, main_classes=main_classes)
                # Check loss
                if it == 1 and "LEN" in strategy:
                    c_loss, arg_max = active_strategy.loss(y_train, return_argmax=True)
                    assert c_loss.sum().item() == 0 or mean(expl_accs) != 1.0, "Error in computing the loss"
                active_idx, active_loss = active_strategy.selection(preds_t, used_idx,
                                                                    n_points, labels=y_t[train_idx],
                                                                    preds_dropout=preds_dropout,
                                                                    clf=net, dataset=train_dataset,
                                                                    main_classes=main_classes)

                if "LEN" in strategy and rand_points > 0:
                    rand_idx, rand_loss = RandomSampling().selection(preds_t, used_idx, rand_points)
                    active_idx = active_idx[:-rand_points] + rand_idx

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
                                     # f"expl: {formulas}, "
                                     # f"expl_acc: {mean(expl_accs) * 100:.2f}, "
                                     f"train acc: {np.mean(df['Train Accuracy']):.2f}, "
                                     f"test acc: {np.mean(df['Test Accuracy']):.2f}, "
                                     f"l: {losses[-1]:.2f}, p: {len(used_idx)}")
                # print("")

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
    # dfs.to_pickle(f"{result_folder}\\results_partial.pkl")
    dfs.to_pickle(f"{result_folder}\\results.pkl")

    mean_auc = dfs.groupby("Strategy").mean().round(2)['Test Accuracy']
    std_auc = dfs.groupby(["Strategy", "Seed"]).mean().groupby("Strategy").std().round(2)['Test Accuracy']
    print("AUC", mean_auc, "+-", std_auc)
    # %% md

    #### Displaying some pictures to visualize training

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
