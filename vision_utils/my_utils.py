import numpy as np
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import vision_utils.transforms as T
from .my_model import my_fasterrcnn_resnet50_fpn


def visualize_boxes(image, prediction, names, show=True):
	from PIL import ImageDraw
	im = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
	boxes = prediction['boxes'].cpu().numpy()
	labels = prediction['labels'].cpu().numpy()
	if "score" in prediction:
		scores = prediction['scores'].cpu().numpy()
	else:
		scores = np.ones_like(labels)
	names = [names[label] for label in labels]
	im_draw = ImageDraw.Draw(im)
	for box, score, name in zip(boxes, scores, names):
		if score > 0.5:
			im_draw.rectangle(box)
			corner = tuple(box[:2] + np.asarray([0, -10]))
			im_draw.text(corner, f"{name}: {score}")
	if show:
		im.show()
	return im


def get_transform(train: bool = True):
	"""
	Helper functions for data augmentation / transformation, which leverages the functions in `refereces/detection`
	"""

	transforms = []
	# converts the image, a PIL image, into a PyTorch Tensor
	transforms.append(T.ToTensor())
	if train:
		# during training, randomly flip the training images
		# and ground-truth for data augmentation
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
	# load an object recognizer model pre-trained on COCO
	model = my_fasterrcnn_resnet50_fpn(pretrained=True)
	# model = torchvision.models.detection.my_fasterrcnn_resnet50_fpn(pretrained=True)
	# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

	# get the number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model