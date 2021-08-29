import torch
from sklearn.metrics import f1_score


class MultiLabelAccuracy():
	def __init__(self, n_classes):
		"""
		Multi-label accuracy metric.
		It calculates the accuracy per class and then it normalize by the number of classes

		:param n_classes: number of classes
		"""
		self.n_classes = n_classes

	def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
		assert predictions.shape[1] == self.n_classes, f"Shape of prediction tensor unexpected {predictions.shape}"
		assert labels.shape[1] == self.n_classes, f"Shape of prediction labels unexpected {labels.shape}"
		accuracy = (predictions > 0.5).eq(labels).sum().item() / labels.shape[0] * 100 / self.n_classes
		return accuracy


class F1:
	def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
		if predictions.dtype == torch.float:
			predictions = predictions > 0.5
		if labels.dtype == torch.float:
			labels = labels > 0.5
		f1_value = f1_score(predictions.cpu().numpy(), labels.cpu().numpy(), average='macro') * 100
		return f1_value
