# %% md

# Constrained Active Learning - Experiment on the PascalPart Object Recognition problem

# %% md

### Importing libraries

# %%
from datetime import datetime

if __name__ == "__main__":

    # %matplotlib inline
    # %autosave 10
    import os

    gpu_id = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    import tqdm

    from kal.vision_utils.pytorchyolo.utils.loss import compute_loss
    import pandas as pd
    from torch.optim import Optimizer

    from kal.vision_utils.pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS, DEFAULT_TRANSFORMS
    from kal.vision_utils.pytorchyolo.utils.datasets import ListDataset
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm, trange
    import torch.multiprocessing
    from typing import Union, Tuple
    from torch import Tensor

    import kal.vision_utils.vis_utils as vis_utils
    import kal.vision_utils.my_utils as my_utils

    from vision_utils.create_custom_model import create_custom_model_configuration_file
    from vision_utils.download_yolo_weights import download_yolo_weights
    from kal.vision_utils.pytorchyolo.models import load_model
    from kal.vision_utils.pytorchyolo.test import print_eval_stats
    from kal.vision_utils.pytorchyolo.utils.parse_config import parse_data_config
    from kal.vision_utils.pytorchyolo.utils.utils import load_classes, ap_per_class, \
        get_batch_statistics, non_max_suppression, xywh2xyxy
    from kal.active_strategies import SUPERVISED, KAL, RANDOM, SAMPLING_STRATEGIES
    from data.DogvsPerson import create_dogvsperson_dataset, \
        bad_targets_idx
    from data.DogvsPerson import n_samples as tot_points
    from kal.knowledge import DogvsPersonLoss

    torch.multiprocessing.set_sharing_strategy('file_system')
    plt.rc('animation', html='jshtml')
    sns.set_theme(style="whitegrid", font="Times New Roman")

    dataset_name = "dogvsperson"
    model_cfg_file = "dogvsperson_yolo.cfg"
    model_folder = os.path.join("models", dataset_name)
    result_folder = os.path.join("results", dataset_name)
    image_folder = os.path.join("images", dataset_name)
    data_folder = os.path.join("..", "data")
    # log_folder = os.path.join("logs", datetime.datetime.now()
    #                           .strftime("%d_%m_%Y__%H_%M_%S"))

    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    if not os.path.isdir(image_folder):
        os.makedirs(image_folder)
    # if not os.path.isdir(log_folder):
    #     os.makedirs(log_folder)

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # %% md

    ### Setting some hyperparameters for the active learning strategies

    # %%
    debug = False
    verbose = False
    load = True

    if debug:
        tot_points = 1000
        first_points = 5
        n_points = 5
        n_iterations = 5
        epochs = 10
        n_cpu = 0
    else:
        first_points = 1000
        n_points = 500
        n_iterations = 4
        epochs = 100
        n_cpu = 8

    seeds = range(3)
    learning_rate = 1/3 * 1e-3
    strategies = [KAL, SUPERVISED, RANDOM]

    dev = f"cuda" if torch.cuda.is_available() else "cpu"
    mini_batch_size = 8 if torch.cuda.is_available() else 4
    iou_thres = 0.5
    conf_thres = 0.01
    nms_thres = 0.5

    for k, v in [*locals().items()][-16:]:
        print(f"{k}: {v}")

    # %% md

    ### Create Dogvsperson dataset

    # %%
    all_idx = np.arange(tot_points)
    bad_targets_idx = [idx for idx in bad_targets_idx if idx in all_idx]
    # bad_targets_idx = []

    # Some idx labels are not consistent with the ground truth knowledge, therefore we eliminate them.
    # We saved bad idx in file so that we don't have to recalculate them all the times
    all_idx = np.asarray(list(set(all_idx.tolist()) - set(bad_targets_idx)))
    tot_points -= len(bad_targets_idx)

    train_idx = np.random.choice(all_idx, int(len(all_idx)*0.9), replace=False)
    test_idx = np.asarray(list(set(all_idx) - set(train_idx)))
    data_config_file = create_dogvsperson_dataset(data_folder, train_idx, test_idx)
    data_config = parse_data_config(data_config_file)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    data_folder = os.path.dirname(train_path)
    class_names = load_classes(data_config["names"])
    n_classes = len(class_names)
    print("Number of classes:", n_classes)
    print("Number of samples:", tot_points)
    print("Number of starting train examples:", first_points)
    # %% md

    ### Creating model

    # %%

    weights = download_yolo_weights(model_folder)
    create_custom_model_configuration_file(n_classes, model_folder, model_cfg_file,
                                           mini_batch_size, learning_rate, override=True)
    model = load_model(os.path.join(model_folder, model_cfg_file), weights)

    KLoss = DogvsPersonLoss

    # %% md

    ### Creating data loader for the Dogvsperson dataset

    # %%
    train_dataset = ListDataset(
        train_path,
        img_size=model.hyperparams['height'],
        multiscale=False,
        transform=AUGMENTATION_TRANSFORMS
    )
    test_dataset = ListDataset(
        valid_path,
        img_size=model.hyperparams['height'],
        multiscale=False,
        transform=DEFAULT_TRANSFORMS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )

    #%% md

    ### Visualizing a sample training data with the corresponding bounding boxes

    #%%

    _, image, target = test_dataset[12]
    im = my_utils.visualize_boxes_yolo(image, target, class_names)
    print(target[0])
    im.save(os.path.join(image_folder, "sample.png"))

    #%% md

    ### Defining constraints as product t-norm of the FOL rule expressing by the structured knowledge of Dogvsperson classes
    # In order to calculate the t-norm we first need to convert the detection results to a class-wise multi-label prediction.
    # Notice: some of the data in the dataset may be inconsistent with the constraints, but we have already remove them

    #%%


    def convert_detection_to_prediction(detections: list,
                                        batch_size: int,
                                        ):
        # Calculating how many samples have been predicted.
        # Last batch may be partially empty.
        n_detections = (len(detections)-1) * batch_size + \
                       len(detections[-1])
        ml_preds = torch.zeros(n_detections, n_classes)
        for k, batch_detect in enumerate(detections):
            for j, sample_detect in enumerate(batch_detect):
                row_idx = k * batch_size + j
                for c in range(n_classes):
                    predicted_class = sample_detect[:, -1] == c
                    if predicted_class.any():
                        score = torch.max(sample_detect[predicted_class, 4])
                        assert 1 >= score >= 0, f"Problem with defining score {score}"
                        ml_preds[row_idx, c] = score

                assert ml_preds[row_idx].sum() < n_classes, \
                    "Error in calculating predictions"
                if ml_preds[row_idx].sum() == 0.0:
                    print("No class predicted for sample", row_idx)

        return ml_preds

    targets = []
    file_idx = []
    for i, data in enumerate(test_loader):
        targets.append(data[2])
        file_idx.extend(data[0])
        if i == 10:
            break
    predictions = []
    for target in targets:
        batch_predictions = []
        for i in range(int(target[:, 0].max() + 1)):
            sample_idx = target[:, 0] == i
            sample_target = target[sample_idx, :]
            sample_prediction = torch.cat([sample_target[:, 2:6],
                                           torch.ones(sample_target.shape[0], 1),
                                           sample_target[:, 1:2]], dim=1)
            batch_predictions.append(sample_prediction)
        predictions.append(batch_predictions)

    predictions = convert_detection_to_prediction(predictions, mini_batch_size)
    cons_loss = KLoss()(predictions, labels=True)
    bad_targets_idx = torch.where(cons_loss > 0.1)[0]
    assert bad_targets_idx.shape[0] == 0, f"Bad target idx {bad_targets_idx}"

    sorted_cons_loss = cons_loss.sort()[0].cpu().numpy()
    sns.scatterplot(x=[*range(len(cons_loss))], y=cons_loss)
    plt.show()

    #%% md

    ### Checking outputs before training

    #%%


    def evaluate(net, dataset: Union[Subset, ListDataset],
                 evaluate_loss=False, verb=False):
        from copy import deepcopy
        if isinstance(dataset, Subset):
            dataset.dataset = deepcopy(dataset.dataset)
            dataset.dataset.transform = DEFAULT_TRANSFORMS
        else:
            dataset = deepcopy(dataset)
            dataset.transform = DEFAULT_TRANSFORMS

        data_loader = DataLoader(
            dataset,
            batch_size=mini_batch_size,
            shuffle=False,
            num_workers=n_cpu,
            pin_memory=True,
            collate_fn=test_dataset.collate_fn,
        )
        img_size = net.hyperparams['height']

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        outputs = []
        loss = torch.as_tensor([]).to(dev)
        pbar = tqdm(data_loader, desc="Validating") if verb else data_loader
        for _, imgs, batch_labels in pbar:

            batch_labels = batch_labels.to(dev)
            imgs = imgs.to(dev)

            # Extract labels
            labels += batch_labels[:, 1].tolist()
            # Rescale target
            batch_labels[:, 2:] = xywh2xyxy(batch_labels[:, 2:])
            batch_labels[:, 2:] *= img_size

            assert (batch_labels[:, 2:] <= img_size).all() \
                   and (batch_labels[:, 2:] >= 0).all(), \
                   "Error in the labels, some are > image or < 0"

            net.to(dev)
            net.eval()  # Set model to evaluation mode
            with torch.no_grad():
                batch_outputs = net(imgs)
                batch_outputs = non_max_suppression(batch_outputs,
                                                    conf_thres=conf_thres,
                                                    iou_thres=nms_thres)
                if evaluate_loss:
                    net.train()
                    batch_outputs_train = net(imgs)
                    batch_loss = compute_loss(batch_outputs_train, batch_labels,
                                              net, reduction="none")[0]
                    loss = torch.cat([loss, batch_loss])
            sample_metrics += get_batch_statistics(batch_outputs,
                                                   batch_labels,
                                                   iou_threshold=iou_thres)

            outputs.append(batch_outputs)
        if len(sample_metrics) == 0:  # No detections over whole validation set.
            print("---- No detections over whole validation set ----")
            return None

        if verb:
            pbar.close()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate([y.cpu() for y in x], 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)

        print_eval_stats(metrics_output, class_names, verb)

        AP = metrics_output[2]

        if evaluate_loss:
            return AP.mean(), outputs, loss.cpu()
        return AP.mean()

    def visualize_data_loss(net, dataset: ListDataset, idx: np.ndarray = None) \
            -> [float, Tensor, Tensor, Tensor, Tensor]:
        acc, detects, s_loss = evaluate(net, dataset,
                                        evaluate_loss=True, verb=verbose)
        p_t = convert_detection_to_prediction(detects, mini_batch_size)
        c_loss, arg_m = KLoss()(p_t, return_arg_max=True)
        c_loss = c_loss.cpu()

        if idx is None:
            sns.scatterplot(x=c_loss.numpy(), y=s_loss.numpy())
        else:
            u_idx = np.zeros_like(s_loss.numpy())
            u_idx[idx] = 1
            sns.scatterplot(x=c_loss.numpy(), y=s_loss.numpy())
        plt.yscale("log")
        plt.title("Supervision vs Constraint Loss")
        plt.show()
        print(f"mAP: {acc}, Loss: {s_loss.mean()}")
        return acc, p_t, s_loss, c_loss, arg_m


    # visualize_data_loss(model, valid_loader)

    #%% md

    ## Few epochs with n randomly selected data

    #%%


    def train_one_epoch(net, train_loader: DataLoader, epoch: int,
                        optim: Optimizer, verb: int = True):

        # Set net to training mode
        net.train()
        epoch_l = torch.tensor([0.])
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}") if verb \
            else train_loader

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(train_loader) - 1)

            lr_scheduler = vis_utils.warmup_lr_scheduler(optim, warmup_iters, warmup_factor)

        for batch_i, (_, imgs, labels) in enumerate(pbar):

            imgs = imgs.to(dev, non_blocking=True)
            labels = labels.to(dev)

            outputs = net(imgs)

            loss, loss_components = compute_loss(outputs, labels, net)
            loss.backward()
            epoch_l += loss.detach().cpu()

            # Run optimizer
            optim.step()
            # Reset gradients
            optim.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

            net.seen += imgs.size(0)

        epoch_l = epoch_l.item() / len(train_loader)

        if verb:
            pbar.close()

        return epoch_l


    def train_loop(net, dataset: ListDataset, idx: np.ndarray, eps: int,
                   visualize_loss: bool = False, verb=True, evaluate_train=False):
        train_set = Subset(dataset, idx)
        train_loader = DataLoader(
            train_set,
            batch_size=mini_batch_size,
            shuffle=True,
            num_workers=n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        # Create optimizer
        parameters = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=epochs//3,
                                                       gamma=1/3)

        l_train = []
        best_l = 1000
        m_aps = [0., ]
        pbar = trange(eps, desc="Training")
        temp_file = os.path.join(model_folder, "temp.pt")
        for epoch in range(eps):
            epoch_l = train_one_epoch(net, train_loader, epoch,
                                      optimizer, verb)
            l_train.append(epoch_l)
            lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]

            # Save best model looking at the loss
            if epoch_l < best_l and epoch > eps // 2:
                best_l = epoch_l
                torch.save(net.state_dict(), temp_file)

            # Evaluate model on training data (skip first epoch too slow)
            if evaluate_train and (epoch > 0):
                AP = evaluate(net, train_set, verb=False)
                m_aps.append(AP)
            pbar.update()
            pbar.set_postfix({
                "Lr": lr,
                "Loss": epoch_l,
                "Best Epoch": l_train.index(best_l) if epoch > eps // 2 else None,
                "AP": AP if evaluate_train and epoch > 0 else None
            })

        net.load_state_dict(torch.load(temp_file))
        pbar.close()

        if visualize_loss:
            sns.lineplot(data=np.asarray(l_train))
            sns.lineplot(data=np.asarray(m_aps))
            plt.ylabel("Loss/AP"), plt.xlabel("Epochs"), plt.yscale("log")
            plt.title("Training loss variations in function of the epochs")
            plt.show()

        return l_train


    for seed in seeds:
        df_file = os.path.join(result_folder, f"metrics_{KAL}_strategy_{seed}_seed_"
                                              f"{first_points}_points.pkl")
        model_file = os.path.join(model_folder, f"model_{KAL}_strategy_{seed}_seed_"
                                                f"{first_points}_points.pt")

        network = load_model(os.path.join(model_folder, model_cfg_file), weights)
        if os.path.exists(model_file) and os.path.exists(df_file) and load:
            print(f"Already trained {model_file} - {datetime.now()}")
            continue
        print(f"Training {model_file} - {datetime.now()}")
        first_idx = np.random.randint(0, len(train_dataset), first_points)
        train_loss = train_loop(network, train_dataset, first_idx,
                                epochs, visualize_loss=True, verb=verbose)
        torch.save(network.state_dict(), model_file)

        test_acc = evaluate(network, test_dataset, verb=verbose)
        train_stats = visualize_data_loss(network, train_dataset, first_idx)
        train_acc, preds_t, sup_loss, cons_loss, arg_max = train_stats

        d = {
            "strategy": [KAL],
            "seed": [seed],
            "iteration": [0],
            "active_idx": [first_idx],
            "used_idx": [first_idx],
            "predictions": [preds_t.cpu().numpy()],
            "test accuracy": [test_acc],
            "train accuracy": [test_acc],
            "train_loss": [train_loss],
            "constraint_loss": [cons_loss.cpu().numpy()],
            "arg_max": [arg_max],
            "supervision_loss": [sup_loss.cpu().numpy()]
        }
        pd.DataFrame(d).to_pickle(df_file)

    # %%

    network = load_model(os.path.join(model_folder, model_cfg_file), weights)

    dfs = []
    for strategy in strategies:
        active_strategy = SAMPLING_STRATEGIES[strategy](k_loss=KLoss, main_classes=[0, 1])

        for seed in seeds:
            for it in range(0, n_iterations + 1):

                used_points = first_points + it * n_points

                df_file = os.path.join(result_folder, f"metrics_{strategy}_strategy_{seed}_seed_"
                                                      f"{used_points}_points.pkl")
                model_file = os.path.join(model_folder, f"model_{strategy}_strategy_{seed}_seed_"
                                                        f"{used_points}_points.pt")
                if it == 0:
                    df_file = os.path.join(result_folder, f"metrics_{KAL}_strategy_{seed}_seed_"
                                                          f"{used_points}_points.pkl")
                    model_file = os.path.join(model_folder, f"model_{KAL}_strategy_{seed}_seed_"
                                                            f"{used_points}_points.pt")

                if os.path.exists(df_file) and os.path.exists(model_file) and (load or it == 0):
                    print(f"Already trained {model_file} - {datetime.now()}")
                    df = pd.read_pickle(df_file)
                    if it == 0:
                        df['strategy'] = strategy
                    dfs.append(df)

                    used_idx = torch.as_tensor(df['used_idx'][0])
                    sup_loss = torch.as_tensor(df['supervision_loss'][0])
                    preds_t = torch.as_tensor(df['predictions'][0])
                    cons_loss, arg_max = KLoss()(preds_t, return_arg_max=True)
                    # cons_loss = torch.as_tensor(dfs['constraint_loss'].iloc[-1])
                    # arg_max = dfs['arg_max'].iloc[-1]
                    available_idx = list({*range(tot_points)} - set(used_idx))

                    # network.load_state_dict(torch.load(model_file, map_location=dev))
                    network = load_model(os.path.join(model_folder, model_cfg_file), weights)

                    accuracy = df['test accuracy'][0]
                    print(f"mAP {accuracy}")

                    continue
                elif it == 0:
                    raise RuntimeError("Error in loading first dataframe")

                print(f"Training {model_file} - {datetime.now()}")

                active_idx, active_loss = active_strategy.selection(preds_t, used_idx,
                                                                    n_points,
                                                                    dataset=train_dataset)

                used_idx = np.append(used_idx, active_idx)

                assert len(used_idx) == used_points, "Error in selecting points"

                train_loss = train_loop(network, train_dataset, used_idx, epochs, verb=verbose)
                torch.save(network.state_dict(), model_file)

                test_accuracy = evaluate(network, test_dataset, verb=verbose)
                train_accuracy, detection, sup_loss = evaluate(network, train_dataset,
                                                               evaluate_loss=True, verb=verbose)

                preds_t = convert_detection_to_prediction(detection, mini_batch_size)
                cons_loss, arg_max = KLoss()(preds_t, return_arg_max=True)

                d = {
                    "strategy": [strategy],
                    "seed": [seed],
                    "iteration": [it],
                    "active_idx": [active_idx.copy()],
                    "used_idx": [used_idx.copy()],
                    "predictions": [preds_t.cpu().numpy()],
                    "train accuracy": [train_accuracy],
                    "test accuracy": [test_accuracy],
                    "train_loss": [train_loss],
                    "constraint_loss": [cons_loss.numpy()],
                    "arg_max": [arg_max],
                    "supervision_loss": [sup_loss.numpy()],
                }
                print(f"\nTrain Acc: {train_accuracy:.3f}, Test Acc: {test_accuracy:.3f}, "
                      f"S loss: {sup_loss.mean().item():.2f}, "
                      f"n p: {len(used_idx)}")

                df = pd.DataFrame(d)
                df.to_pickle(df_file)
                dfs.append(df)

            if seed == 0:
                losses = pd.concat(dfs)['train_loss'][pd.concat(dfs)['strategy'] == strategy]
                losses = np.concatenate([*losses])
                sns.lineplot(data=losses)
                plt.yscale("log")
                plt.ylabel("Loss")
                plt.xlabel("Epochs")
                plt.title(f"Training loss variations for {strategy} active learning strategy")
                plt.show()

    dfs = pd.concat(dfs)
    dfs.to_pickle(os.path.join(result_folder, f"metrics.pkl"))

    # %%

    strategy_mappings = {
        RANDOM: "Random",
        "random": "Random",
        KAL: "KAL",
        "constrained": "KAL",
        SUPERVISED: "Supervised",
        "supervised": "Supervised"
    }
    df = pd.read_pickle(os.path.join(result_folder, f"metrics.pkl"))
    df['n_points'] = [len(used) for used in df['used_idx']]
    a = []
    for _, row in df.iterrows():
        a.append(strategy_mappings[row['strategy']])
    df["Strategy"] = a
    df['test accuracy'] = df['test accuracy'] * 100
    df = df.reset_index()

    # %%
    df = df.sort_values(['Strategy'])
    sns.set(style="whitegrid", font_scale=1.2,
            rc={'figure.figsize': (5, 4)})
    sns.lineplot(data=df, x="n_points", y="test accuracy",
                 hue="Strategy", ci=75)
    plt.ylabel("mAP")
    plt.xlabel("Number of points used")
    # plt.ylim([15, 30])
    # plt.xlim([550, 1050])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    # plt.title("Comparison of the accuracies in the various strategy in function of the iterations")
    plt.savefig(os.path.join(image_folder, f"Accuracy.png"), dpi=200)
    plt.show()

    # %%

