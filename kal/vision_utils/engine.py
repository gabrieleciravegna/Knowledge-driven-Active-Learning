from torch.utils.data import Subset
import math
import os
import sys
import time

import pickle
import torch

import torchvision.models.detection.mask_rcnn
from typing import Union, Tuple, List

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import vis_utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, verbose=True):
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    if not verbose:
        sys.stdout = None
        sys.stderr = None

    model.train()
    metric_logger = vis_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', vis_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = vis_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = vis_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, evaluate_loss=False, verbose=True) -> \
        Union[Tuple[float, List],
              Tuple[float, List, torch.Tensor]]:

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    if not verbose:
        sys.stdout = None
        sys.stderr = None

    print("Evaluating")
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = vis_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    load = True
    dataset = data_loader.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    coco_file = os.path.join(dataset.root, "coco_evaluator.pckl")
    if os.path.exists(coco_file) and load:
        with open(coco_file, "rb") as f:
            coco_evaluator = pickle.load(f)
        print("Coco evaluator loaded")
    else:
        print("Creating Coco API for current dataset")
        coco_time = time.time()
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        coco_time = time.time() - coco_time
        with open(coco_file, "wb") as f:
            pickle.dump(coco_evaluator, f)
        print(f"Coco created in {coco_time}s")

    loss_dict = None
    outputs, losses = [], []
    # data_loader.num_workers = 0  # TODO: take out
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        o = model(images, targets)

        if evaluate_loss:
            model.train()
            loss_dict = model(images, targets)
            model.eval()

        model_time = time.time() - model_time
        o = [{k: v.to(cpu_device) for k, v in t.items()} for t in o]

        res = {target["image_id"].item(): output for target, output in zip(targets, o)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if evaluate_loss:
            reduced_loss = sum(loss for loss in loss_dict.values())
            losses.append(reduced_loss)
        outputs.extend(o)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    mAP = coco_evaluator.coco_eval['bbox'].stats[0] * 100
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

    if evaluate_loss:
        losses = torch.stack(losses).cpu()
        return mAP, outputs, losses

    return mAP, outputs


# def predict(model, data_loader, device, print_freq=10, verbose=True):
#     orig_stdout = sys.stdout
#     if not verbose:
#         sys.stdout = None
#
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#
#     for images, targets in metric_logger.log_every(data_loader, print_freq):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         loss_dict = model(images, targets)
#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#
#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#
#     sys.stdout = orig_stdout
#
#     return metric_logger

