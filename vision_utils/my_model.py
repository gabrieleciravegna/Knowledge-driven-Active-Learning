from collections import OrderedDict
from typing import List, Tuple, Optional, Dict

import torch
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class MyFasterRCNN(FasterRCNN):

	def forward(self, images, targets=None):
		# type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
		"""
		Arguments:
			images (list[Tensor]): images to be processed
			targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

		Returns:
			result (list[BoxList] or dict[Tensor]): the output from the model.
				During training, it returns a dict[Tensor] which contains the losses.
				During testing, it returns list[BoxList] contains additional fields
				like `scores`, `labels` and `mask` (for Mask R-CNN models).

		"""
		if self.training and targets is None:
			raise ValueError("In training mode, targets should be passed")
		if self.training:
			assert targets is not None
			for target in targets:
				boxes = target["boxes"]
				if isinstance(boxes, torch.Tensor):
					if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
						raise ValueError("Expected target boxes to be a tensor"
						                 "of shape [N, 4], got {:}.".format(
							boxes.shape))
				else:
					raise ValueError("Expected target boxes to be of type "
					                 "Tensor, got {:}.".format(type(boxes)))

		original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
		for img in images:
			val = img.shape[-2:]
			assert len(val) == 2
			original_image_sizes.append((val[0], val[1]))

		images, targets = self.transform(images, targets)

		# Check for degenerate boxes
		# TODO: Move this to a function
		if targets is not None:
			for target_idx, target in enumerate(targets):
				boxes = target["boxes"]
				degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
				if degenerate_boxes.any():
					# print the first degenerate box
					bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
					degen_bb: List[float] = boxes[bb_idx].tolist()
					raise ValueError("All bounding boxes should have positive height and width."
					                 " Found invalid box {} for target at index {}."
					                 .format(degen_bb, target_idx))

		features = self.backbone(images.tensors)
		if isinstance(features, torch.Tensor):
			features = OrderedDict([('0', features)])
		proposals, proposal_losses = self.rpn(images, features, targets)
		detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
		detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

		losses = {}
		losses.update(detector_losses)
		losses.update(proposal_losses)

		if not self.training and targets is not None:
			return (losses, detections)
		else:
			return self.eager_outputs(losses, detections)


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


def my_fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):

	assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
	# dont freeze any layers if pretrained model or backbone is not used
	if not (pretrained or pretrained_backbone):
		trainable_backbone_layers = 5
	if pretrained:
		# no need to download the backbone if pretrained is set
		pretrained_backbone = False
	backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
	model = MyFasterRCNN(backbone, num_classes, **kwargs)
	if pretrained:
		state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
		                                      progress=progress)
		model.load_state_dict(state_dict)
	return model