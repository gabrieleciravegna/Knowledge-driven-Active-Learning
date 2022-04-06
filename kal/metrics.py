import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder


class Metric:
	def __init__(self):
		pass


class MultiLabelAccuracy(Metric):
	def __init__(self, n_classes):
		"""
		Multi-label accuracy metric.
		It calculates the accuracy per class and then it normalize by the number of classes

		:param n_classes: number of classes
		"""
		super().__init__()
		self.n_classes = n_classes

	def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
		if self.n_classes == 1:
			return (predictions > 0.5).eq(labels).sum().item() / labels.shape[0] * 100

		assert predictions.shape[1] == self.n_classes, \
			f"Unexpected shape of prediction tensor {predictions.shape}"

		if len(labels.squeeze().shape) == 1:
			labels = OneHotEncoder().fit_transform(labels)

		correct_predictions = (predictions > 0.5).eq(labels).sum().item()
		accuracy = correct_predictions / labels.shape[0] / self.n_classes * 100

		return accuracy


class F1(Metric):
	def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
		if predictions.dtype == torch.float:
			predictions = predictions > 0.5
		if labels.dtype == torch.float:
			labels = labels > 0.5
		f1_value = f1_score(predictions.cpu().numpy(), labels.cpu().numpy(),
							average='macro') * 100
		return f1_value
