from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch import nn, Tensor
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import transforms as T


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def convert_yolo_to_coco_targets(targets_yolo: List[torch.Tensor],
                                 images: List[torch.Tensor],
                                 indexes: List[int]) -> List[Dict[str, torch.Tensor]]:
    coco_targets = []
    for target, image, index in zip(targets_yolo, images, indexes):
        img_idx = target[:, 0].to(int)
        labels = target[:, 1].to(int) + 1  # yolo predicts from 0, in coco 0 is the background
        boxes = xywh2xyxy(target[:, 2:])
        boxes[:, 0] = boxes[:, 0] * image.shape[1]
        boxes[:, 1] = boxes[:, 1] * image.shape[2]
        boxes[:, 2] = boxes[:, 2] * image.shape[1]
        boxes[:, 3] = boxes[:, 3] * image.shape[2]
        coco_target = {
            'labels': labels,
            'boxes': boxes,
            'image_id': index
        }
        coco_targets.append(coco_target)

    return coco_targets


def convert_yolo_to_coco_outputs(outputs_yolo: List[torch.Tensor],
                                 images: List[torch.Tensor], rescale: bool = True) \
        -> List[Dict[str, torch.Tensor]]:
    outputs = []
    for output_yolo, image in zip(outputs_yolo, images):
        labels = torch.round(output_yolo[:, 5]).to(int) + 1  # yolo predicts from 0, in coco 0 is the background
        scores = output_yolo[:, 4]
        boxes = output_yolo[:, 0:4]
        if rescale:
            boxes[:, 0] = boxes[:, 0] * image.shape[1]
            boxes[:, 1] = boxes[:, 1] * image.shape[2]
            boxes[:, 2] = boxes[:, 2] * image.shape[1]
            boxes[:, 3] = boxes[:, 3] * image.shape[2]
        output_coco = {
            'labels': labels,
            'scores': scores,
            'boxes': boxes,
        }
        outputs.append(output_coco)

    return outputs


def my_collate_fn(batch):
    img, target, shapes, indexes = zip(*batch)  # transposed
    for i, t in enumerate(target):
        t[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img), target, shapes, indexes


def visualize_boxes_yolo(image, predictions=None, names=None, show=False, scores=None):
    im = transforms.ToPILImage()(image)
    im_draw = ImageDraw.Draw(im)
    if predictions is not None and names is not None:
        if scores is None:
            scores = torch.ones(predictions.shape[0])
        boxes = predictions[:, 2:]
        boxes = xywhn2xyxy(boxes, im.width, im.height).numpy()
        labels = predictions[:, 1].int().numpy()
        for box, label, score in zip(boxes, labels, scores):
            name = names[label]
            if score > 0.5:
                im_draw.rectangle(box)
                corner = tuple(box[:2] + np.asarray([0, +5]))
                im_draw.text(corner, f"{name}: {score}")
    if show:
        im.show()
    return im

def visualize_boxes(image, prediction, names, show=True, add_score=True, save_path=None):
    from PIL import ImageDraw
    im = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    areas = prediction['area'].cpu().numpy()
    if "score" in prediction:
        scores = prediction['scores'].cpu().numpy()
    else:
        scores = np.ones_like(labels)
    im_draw = ImageDraw.Draw(im)
    for box, score, label, area in zip(boxes, scores, labels, areas):
        name = names[label]
        if score > 0.5 and 100 < area < 100000:
            im_draw.rectangle(box)
            corner = tuple(box[:2] + np.asarray([0, +5]))
            text = f"{name}: {score}" if add_score else f"{name}"
            im_draw.text(corner, text)
    if save_path is not None:
        im.save(save_path)
    if show:
        im.show()
    return im


class ResizeAndConvertToYOLO(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.resizer = torchvision.transforms.Resize([size, size])

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Tensor]]:
        resized_image = self.resizer(image)  # TODO: not working, we need to adjust it to allow for batch training
        # resized_image = image
        if target is not None:
            labels = target['labels']
            yolo_box = xyxy2xywh(target['boxes'])  # / self.size
            x = yolo_box[:, 0] / image.shape[1]
            y = yolo_box[:, 1] / image.shape[2]
            w = yolo_box[:, 2] / image.shape[1]
            h = yolo_box[:, 3] / image.shape[2]
            img_idx = torch.zeros_like(labels)
            resized_targets = torch.stack([img_idx, labels, x, y, w, h]).T

            return resized_image, resized_targets

        return resized_image, target


def get_transform(train: bool = True, size=416, random_flip=0.5):
    """
    Helper functions for data augmentation / transformation, which leverages the functions in `refereces/detection`
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor

    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(random_flip))
    # transforms.append(ResizeAndConvertToYOLO(size))
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    # load an object recognizer model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = my_fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
