from torch.utils.data import Subset
import math
import os
import sys
import time

import pickle
import torch

import torchvision.models.detection.mask_rcnn
from typing import Union, Tuple, List

from my_utils import convert_yolo_to_coco_outputs, convert_yolo_to_coco_targets
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import vis_utils
from .yolov3.utils.loss import ComputeLoss
from .yolov3.utils.general import scale_coords, non_max_suppression


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, verbose=True):
    orig_stdout = sys.stdout
    if not verbose:
        sys.stdout = None

    model.train()
    metric_logger = vis_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', vis_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    compute_loss = ComputeLoss(model)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = vis_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets, _, _ in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.stack(images).to(device)
        targets = torch.cat(targets, dim=0).to(device)

        predictions = model(images)
        loss_value, loss_values = compute_loss(predictions, targets)  # loss scaled by batch_size

        # reduce losses over all GPUs for logging purposes
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, loss1=loss_values[0],
                             loss2=loss_values[1], loss3=loss_values[2])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    sys.stdout = orig_stdout

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
    if not verbose:
        sys.stdout = None

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
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
    else:
        coco_time = time.time()
        coco = get_coco_api_from_dataset(dataset.get_coco_compatible_dataset())
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        coco_time = time.time() - coco_time
        with open(coco_file, "wb") as f:
            pickle.dump(coco_evaluator, f)
        print(f"Coco created in {coco_time}s")

    model.eval()
    compute_loss = ComputeLoss(model)
    outputs_l, losses_l = [], []
    # data_loader.num_workers = 0  # TODO: take out
    for images, targets, shapes, index in metric_logger.\
		    log_every(data_loader, 1000//data_loader.batch_size, header):
        images = images.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs, outputs_train = model(images)
        outputs = non_max_suppression(outputs)
        for im, out, s in zip(images, outputs, shapes):
            scale_coords(im.shape[1:], out[:, :4], s[0], s[1])  # native-space pred

        if evaluate_loss:
            for i in range(len(targets)):
                t = targets[i].to(device)
                t[:, 0] = 0  # set as img idx 0 all the targets beacuse we are evaluating one sample at a time
                o_train = [o[i].unsqueeze(dim=0) for o in outputs_train]
                loss_value, _ = compute_loss(o_train, t)  # loss scaled by batch_size
                losses_l.append(loss_value)

        model_time = time.time() - model_time
        coco_outputs = convert_yolo_to_coco_outputs(outputs, images, rescale=False)
        coco_targets = convert_yolo_to_coco_targets(targets, images, index)
        outputs_l.extend([*coco_outputs])

        res = {target["image_id"].item(): output
               for target, output in zip(coco_targets, coco_outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

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

    if evaluate_loss:
        losses_l = torch.stack(losses_l).squeeze().cpu()
        return mAP, outputs_l, losses_l

    return mAP, outputs_l


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

