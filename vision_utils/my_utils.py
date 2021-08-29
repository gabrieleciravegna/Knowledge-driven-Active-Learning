
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import vision_utils.transforms as T


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
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

	# get the number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model