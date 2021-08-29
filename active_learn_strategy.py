import numpy as np
import torch


def cal_selection(not_avail_idx: list, c_loss: torch.Tensor, n_p: int) -> list:
	"""
	Constrained Active learning strategy.
	We take n elements which are the one that most violates the constraints
	and are among available idx

	:param not_avail_idx: unavailable data (already selected)
	:param c_loss: constraint violation calculated for each point
	:param n_p: number of points to select
	:return selected idx
	"""
	c_loss[torch.as_tensor(not_avail_idx)] = -1
	cal_idx = torch.argsort(c_loss, descending=True).tolist()[:n_p]
	return cal_idx


def random_selection(avail_idx: list, n_p: int) -> list:
	"""
	Random Active learning strategy
	Theoretically the worst possible strategy. At each iteration
	we just take n elements randomly

	:param avail_idx: available data (not already selected)
	:param n_p: number of points to select
	:return selected idx
	"""
	random_idx = np.random.choice(avail_idx, n_p).tolist()
	return random_idx


def supervised_selection(not_avail_idx: list, s_loss: torch.Tensor, n_p: int) -> list:
	"""
	Supervised Active learning strategy
	Possibly an upper bound to a learning strategy efficacy (fake, obviously).
	We directly select the point which mostly violates the supervision loss.

	:param not_avail_idx: unavailable data (already selected)
	:param s_loss: supervision violation calculated for each point
	:param n_p: number of points to select
	:return: selected idx
	"""
	s_loss[torch.as_tensor(not_avail_idx)] = -1
	sup_idx = torch.argsort(s_loss, descending=True).tolist()[:n_p]
	return sup_idx


def uncertainty_loss(p: torch.Tensor):
	"""
	We define as uncertainty a metric function for calculating the
	proximity to the boundary (predictions = 0.5).
	In order to be a proper metric function we take the opposite of
	the distance from the boundary mapped into [0,1]
	uncertainty = 1 - 2 * ||preds - 0.5||

	:param p: predictions of the network
	:return: uncertainty measure
	"""
	distance = torch.abs(p - 0.5)
	if len(p.shape) > 1:
		distance = distance.mean(dim=1)
	uncertainty = 1 - 2 * distance
	return uncertainty


def uncertainty_selection(not_avail_idx: list, u_loss: torch.Tensor, n_p: int) -> list:
	"""
	Uncertainty Active learning strategy
	We take n elements which are the ones on which the networks is
	mostly uncertain (i.e. the points lying closer to the decision boundaries).

	:param not_avail_idx: unavailable data (already selected)
	:param s_loss: supervision violation calculated for each point
	:param n_p: number of points to select
	:return selected idx
	"""
	u_loss[torch.as_tensor(not_avail_idx)] = -1
	uncertain_idx = torch.argsort(u_loss, descending=True).tolist()[:n_p]
	return uncertain_idx


SUPERVISED = "supervised"
RANDOM = "random"
CAL = "constrained"
UNCERTAIN = "uncertainty"

