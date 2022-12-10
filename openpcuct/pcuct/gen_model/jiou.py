'''
Adapted from
https://github.com/ZiningWang/Inferring-Spatial-Uncertainty-in-Object-Detection
'''
import numpy as np
import numba
from .vis import generate_sample_points_3d, generate_sample_points_bev

class label_uncertainty_IoU:
	'''
	This class is for calculating the BEV JIoUs.
	'''
	def __init__(self, grid_size=0.1, range=3.0):
		self.grid_size = grid_size
		self.range = range  # sample upto how many times of the size of the car

	def calc_IoU(self, uncertain_label, pred_boxBEVs, sample_grid=0.1):
		from pcuct.ops.jiou.jiou_utils import Jaccard_discrete
		sample_points, _ = self.get_sample_points(uncertain_label)
		px = uncertain_label.sample_prob(sample_points, sample_grid=sample_grid)
		JIoUs = []
		for i in range(len(pred_boxBEVs)):
			py = pred_boxBEVs[i].sample_prob(sample_points, sample_grid=sample_grid)
			if np.sum(py) > 0 and np.sum(px) > 0:
				JIoUs.append(Jaccard_discrete(px, py))
			else:
				JIoUs.append(0)
		return JIoUs

	def get_sample_points(self, uncertain_label):
		return generate_sample_points_bev(
			self.range, self.grid_size, uncertain_label)

class label_uncertainty_IoU_3D(label_uncertainty_IoU):
	'''
	This class is for calculating the 3D JIoUs.
	'''
	def calc_IoU(self, uncertain_label, pred_box3Ds, sample_grid=0.1):
		return super().calc_IoU(uncertain_label, pred_box3Ds, sample_grid)

	def get_sample_points(self, uncertain_label):
		return generate_sample_points_3d(
			self.range, self.grid_size, uncertain_label)

