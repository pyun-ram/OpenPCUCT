import torch
import numpy as np
from typing import *

from . import jiou_bev_cpp
from . import jiou_3d_cpp

def jiou_eval_bev(
	boxes:np.ndarray,
	ucts:Dict[str,np.ndarray],
	qboxes:np.ndarray,
	qucts:Dict[str, np.ndarray],
	sample_range: float,
	sample_grid_size: float,
	use_mean:bool=False)->np.ndarray:
	'''
	Compute BEV JIoUs between boxes and qboxes.
	Args:
		boxes (N, 5): BEV boxes [xc, yc, l, w, ry]
		ucts (N): predictive distribution
			{'mean': (1,6)/(1,5),
			 'cov': (6,6)/(5,5)}
		qboxes (M, 5): [xc, yc, l, w, ry] (query)
		qucts (M): predictive distribution (query)
			{'mean': (1,6)/(1,5),
			 'cov': (6,6)/(5,5)}
		sample_range:
		sample_grid_size:
		use_mean: if True, will use ucts['mean] instead of <boxes>.
	Returns:
		jiou (N, M)
	'''
	from ...gen_model.label import uncertain_label_BEV
	def encode_pdist(ucts):
		covs_list, means_list, _masks_list = [], [], []
		for uct in ucts:
			cov = uct['cov'].reshape(1, -1)
			mean = uct['mean'].reshape(1, -1)
			if cov.size == 36:
				_mask = 0
			elif cov.size == 25:
				_mask = 1
				tmp = np.zeros(36)
				tmp[:25] = cov.reshape(-1)
				cov=tmp.reshape(1, -1)
			else:
				err_msg = f"Unreconized shape {cov.shape}"
				raise RuntimeError(err_msg)
			if mean.size == 6:
				mean = uncertain_label_BEV.feature0_to_x0(mean.reshape(-1))
				mean = mean.reshape(1, -1)
			elif mean.size == 5:
				pass
			else:
				err_msg = f"Unreconized shape {mean.shape}"
				raise RuntimeError(err_msg)
			covs_list.append(cov)
			means_list.append(mean)
			_masks_list.append(_mask)
		covs = torch.from_numpy(np.concatenate(covs_list, axis=0)).float()
		means = torch.from_numpy(np.concatenate(means_list, axis=0)).float()
		_masks = torch.FloatTensor(_masks_list)
		return covs, means, _masks

	pdist_covs, pdist_means, _masks = encode_pdist(ucts)
	qpdist_covs, qpdist_means, _qmasks = encode_pdist(qucts)
	boxes_ts = torch.from_numpy(boxes).float()
	qboxes_ts = torch.from_numpy(qboxes).float()
	jiou_ts = jiou_bev_cpp.jiou_eval_bev_cpp(
			boxes_ts if not use_mean else pdist_means, # (N,5)
			pdist_covs, # (N,36)
			_masks, # (N,)
			qboxes_ts if not use_mean else qpdist_means, # (M,5)
			qpdist_covs, # (M,36)
			_qmasks, # (M,)
			sample_range,
			sample_grid_size)
	return jiou_ts.numpy()

def jiou_eval_3d(
	boxes:np.ndarray,
	ucts:Dict[str, np.ndarray],
	qboxes:np.ndarray,
	qucts:List[Dict[str, np.ndarray]],
	sample_range:float,
	sample_grid_size:float,
	use_mean:bool=False)->List[float]:
	'''
	Compute 3D JIoUs between boxes and qboxes.
	Args:
		boxes (N,7): [xc,yc,zc,l,w,h,ry]
		ucts (N): {'mean': (1,8)/(1,7),
			       'cov' : (8,8)/(7,7)}
		qboxes (M,7): [xc,yc,zc,l,w,h,ry] (query)
		quct (M): [Dict] {'mean': (1,8)/(1,7), (query)
						  'cov' :(8,8)/(7,7)}
	Returns
		jious (N,M): 3D JIoUs
	'''
	from ...gen_model.label import uncertain_label_3D
	def encode_pdist(ucts):
		covs_list, means_list, _masks_list = [], [], []
		for uct in ucts:
			cov = uct['cov'].reshape(1, -1)
			mean = uct['mean'].reshape(1, -1)
			if cov.size == 8*8:
				_mask = 0
			elif cov.size == 7*7:
				_mask = 1
				tmp = np.zeros(8*8)
				tmp[:7*7] = cov.reshape(-1)
				cov=tmp.reshape(1, -1)
			else:
				err_msg = f"Unreconized shape {cov.shape}"
				raise RuntimeError(err_msg)
			if mean.size == 8:
				mean = uncertain_label_3D.feature0_to_x0(mean.reshape(-1))
				mean = mean.reshape(1, -1)
			elif mean.size == 7:
				pass
			else:
				err_msg = f"Unreconized shape {mean.shape}"
				raise RuntimeError(err_msg)
			covs_list.append(cov)
			means_list.append(mean)
			_masks_list.append(_mask)
		covs = torch.from_numpy(np.concatenate(covs_list, axis=0)).float()
		means = torch.from_numpy(np.concatenate(means_list, axis=0)).float()
		_masks = torch.FloatTensor(_masks_list)
		return covs, means, _masks
	jiou_3d_cpp.set_num_threads(
		jiou_3d_cpp.get_max_threads()
	)
	pdist_covs, pdist_means, _masks = encode_pdist(ucts)
	qpdist_covs, qpdist_means, _qmasks = encode_pdist(qucts)
	boxes_ts = torch.from_numpy(boxes).float()
	qboxes_ts = torch.from_numpy(qboxes).float()
	jiou_ts = jiou_3d_cpp.jiou_eval_3d_cpp(
			boxes_ts if not use_mean else pdist_means, # (N,7)
			pdist_covs, # (N,64)
			_masks, # (N,)
			qboxes_ts if not use_mean else qpdist_means, # (M,7)
			qpdist_covs, # (M,64)
			_qmasks, # (M,)
			sample_range,
			sample_grid_size)
	return jiou_ts.numpy()

def jacobian_z_normed_x0_bev(mean, z_normed):
	'''
	Compute Jacobian matrix 
	\frac{\partial [x, y]}
	{\partial{xc, yc, l, w, ry}}
	Args:
		z_normed (np.ndarray (N, 2))
	Returns:
		J (np.ndarry (N, 2, 5))
	Notes:
	The relationship is:
	[x] = [cosry*l -sinry*w] [z_normed_0] + [x_c]
	[y]   [sinry*l  cosry*w] [z_normed_1]   [y_c]
	The Jacobian J_xy_x0 is:
	\begin{bmatrix}
	\frac{\partial x}{\partial x_c} & \frac{\partial x}{\partial y_c} & \frac{\partial x}{\partial l} & \frac{\partial x}{\partial w} & \frac{\partial x}{\partial \theta}\\ 
	\frac{\partial y}{\partial x_c} & \frac{\partial y}{\partial y_c} & \frac{\partial y}{\partial l} & \frac{\partial y}{\partial w} & \frac{\partial y}{\partial \theta} \end{bmatrix}
	'''
	err_msg = "Wrong shape."
	assert mean.size == 5, err_msg
	assert z_normed.shape[-1] == 2, err_msg
	rtn = jiou_bev_cpp.jacobian_z_normed_x0_cpp(
		mean.reshape(-1).astype(np.double),
		z_normed.astype(np.double))
	return rtn.astype(mean.dtype)

def jacobian_z_normed_feat0_bev(z_normed):
	'''
        Compute Jacobian matrix 
        \frac{\partial [x, y]}
        {\partial{xc1, xc2, cos(ry)*l, sin(ry)*l, cos(ry)*w, sin(ry)*w}}
        Args:
            z_normed (np.ndarray (N, 2))
        Returns:
            J (np.ndarry (N, 2, 6))
        Notes:
        The relationship is:
        [x] = [cosry*l -sinry*w] [z_normed_0] + [x_c]
        [y]   [sinry*l  cosry*w] [z_normed_1]   [y_c]
        The Jacobian J_xy_feature0 is:
        \begin{bmatrix}
        \frac{\partial x}{\partial x_c} & \frac{\partial x}{\partial y_c} & \frac{\partial x}{\partial lcos\theta} & \frac{\partial x}{\partial lsin\theta} & \frac{\partial x}{\partial wcos\theta} & \frac{\partial x}{\partial wsin\theta}\\ 
        \frac{\partial y}{\partial x_c} & \frac{\partial y}{\partial y_c} & \frac{\partial y}{\partial lcos\theta} & \frac{\partial y}{\partial lsin\theta} & \frac{\partial y}{\partial wcos\theta} & \frac{\partial y}{\partial wsin\theta}\end{bmatrix}
	'''
	err_msg = "Wrong shape."
	assert z_normed.shape[-1] == 2, err_msg
	rtn = jiou_bev_cpp.jacobian_z_normed_feat0_cpp(
		z_normed.astype(np.double))
	return rtn.astype(z_normed.dtype)

def sample_prob_bev(points, mean, cov, sample_grid):
	'''
	Sample probability given points.
	Args:
		points (np.ndarray (N, 2)): 2D coordinates
		sample_grid (float): resolution of the sample grid.
	Returns:
		probs (np.ndarray (N)): probability of points in the posterior distribution.
	'''
	err_msg = "Wrong shape."
	assert points.shape[-1] == 2, err_msg
	assert mean.shape == (5,)
	assert cov.shape == (5,5) or cov.shape == (6,6)
	rtn = jiou_bev_cpp.sample_prob_cpp(
		points.astype(np.double),
		mean.astype(np.double),
		cov.astype(np.double),
		float(sample_grid))
	return rtn.astype(points.dtype)

def calc_uncertainty_points_bev(z_normed, mean, cov):
	'''
	Propagate the uncertainty from label posterior distribution to 2D points.
	Args:
		z_normed (np.ndarray (N, 2))
	Returns:
		mean (np.ndarray (N, 2))
		cov (np.ndarray (N, 2, 2))
	'''
	err_msg = "Wrong shape."
	assert z_normed.shape[-1] == 2, err_msg
	N = z_normed.shape[0]
	means_out = np.zeros_like(z_normed, np.double)
	covs_out = np.zeros((N, 2, 2), np.double)
	jiou_bev_cpp.calc_uncertainty_points_cpp(
		z_normed.astype(np.double),
		mean.astype(np.double),
		cov.astype(np.double),
		means_out, covs_out)
	dtype = z_normed.dtype
	return means_out.astype(dtype), covs_out.astype(dtype)

def jacobian_z_normed_x0_3d(mean, z_normed):
	'''
	Compute Jacobian matrix
	\frac{\partial [x, y, z]}
	{\partial{xc, yc, zc, l, w, h, ry}}
	Args:
		mean (np.ndarray (7))
		z_normed (np.ndarray (N,3))
	Returns:
		J (np.ndarray (N, 3, 7))
	Notes:
	The relationship is
	[x] = [lcosry, -wsinry, 0] [z_norm0] + [xc]
	[y]   [lsinry, wcosry , 0] [z_norm1]   [yc]
	[z]   [0     , 0      , h] [z_norm2]   [zc]
	'''
	err_msg = "Wrong shape."
	assert mean.size == 7, err_msg
	assert z_normed.shape[-1] == 3, err_msg
	rtn = jiou_3d_cpp.jacobian_z_normed_x0_cpp(
		mean.reshape(-1).astype(np.double),
		z_normed.astype(np.double))
	return rtn.astype(mean.dtype)

def jacobian_z_normed_feat0_3d(z_normed):
	'''
	Compute Jacobian matrix
	\frac{\partial [x, y, z]}
	{\partial{xc1, yc1, zc1, lcosry, lsinry, wcosry, wsinry, h}}
	Args:
		z_normed (np.ndarray (N,3))
	Returns:
		J (np.ndarray (N, 3, 8))
	Notes:
	The relationship is
	[x] = [lcosry, -wsinry, 0] [z_norm0] + [xc]
	[y]   [lsinry, wcosry , 0] [z_norm1]   [yc]
	[z]   [0     , 0      , h] [z_norm2]   [zc]
	'''
	err_msg = "Wrong shape."
	assert z_normed.shape[-1] == 3, err_msg
	rtn = jiou_3d_cpp.jacobian_z_normed_feat0_cpp(
		z_normed.astype(np.double))
	return rtn.astype(z_normed.dtype)

def sample_prob_3d(points, mean, cov, sample_grid):
	'''
	Sample probability given points.
	Args:
		points(np.ndarray (N, 3)): 3D coordinates
		mean(np.ndarray (7))
		cov(np.ndarray (7,7)/(8,8))
		sample_grid (float): resolution of the sample grid.
	Returns:
		probs (np.ndarray (N)): probability of points in the posterior
		distribution.
	'''
	err_msg = "Wrong shape."
	assert points.shape[-1] == 3, err_msg
	assert mean.shape == (7,)
	assert cov.shape == (7,7) or cov.shape == (8,8)
	jiou_3d_cpp.set_num_threads(
		jiou_3d_cpp.get_max_threads()
	)
	rtn = jiou_3d_cpp.sample_prob_cpp(
		points.astype(np.double),
		mean.astype(np.double),
		cov.astype(np.double),
		float(sample_grid))
	return rtn.astype(points.dtype)

def calc_uncertainty_points_3d(z_normed, mean, cov):
	'''
	Propagate the uncertainty from label posterior distribution to 3D points.
	Args:
		z_normed (np.ndarray (N, 3))
	Returns:
		means_out (np.ndarray (N, 3))
		covs_out (np.ndarray (N, 3, 3))
	'''
	err_msg = "Wrong shape."
	assert z_normed.shape[-1] == 3, err_msg
	N = z_normed.shape[0]
	means_out = np.zeros_like(z_normed, np.double)
	covs_out = np.zeros((N, 3, 3), np.double)
	jiou_3d_cpp.calc_uncertainty_points_cpp(
		z_normed.astype(np.double),
		mean.astype(np.double),
		cov.astype(np.double),
		means_out, covs_out)
	dtype = z_normed.dtype
	return means_out.astype(dtype), covs_out.astype(dtype)

def Jaccard_discrete(px, py):
	'''
	Calculate Jaccard discrete.
	Args:
		px (np.ndarray (N,))
		py (np.ndarray (N,))
	Returns:
		jaccard_index (float)
	'''
	rtn = jiou_3d_cpp.jaccard_discrete_cpp(
		px.astype(np.double),
		py.astype(np.double))
	return float(rtn)
