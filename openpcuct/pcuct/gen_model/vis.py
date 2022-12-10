'''
Adapted from
https://github.com/ZiningWang/Inferring-Spatial-Uncertainty-in-Object-Detection
'''
import numpy as np
from typing import *
from matplotlib import pyplot as plt
from copy import deepcopy

from det3.visualizer.vis import BEVImage
from det3.dataloader.kittidata import KittiCalib, KittiObj
from det3.dataloader.udidata import UdiObj, UdiCalib
from .label import uncertain_label_3D, uncertain_label_BEV
from .infer import label_inference_BEV, label_inference_3D

def get_sample_points_with_probs_bev(
        mean, cov,
        x_range, y_range,
        sample_range, sample_grid_size):
    '''
    Compute sample points and their probabilities given the mean and cov.
    Args:
        mean (np.ndarray, (1,5))
        cov (np.ndarray, (5,5)/(6,6))
        x_range (Tuple[float]): xmin, xmax
        y_range (Tuple[float]): ymin, ymax
        sample_range (float): sampling range in the hidden space of the generative model.
        sample_grid_size (float): sampling grid size in the hidden space of the generative model.
    Returns:
        sample_points (np.ndarray, (N, 2))
        px (np.ndarray, (N))
        px_shape (dim_x, dim_y): N = dim_x * dim_y
    '''
    min_x, max_x = x_range
    min_y, max_y = y_range
    uct_label = uncertain_label_BEV(mean)
    uct_label.set_posterior(mean, cov)
    sample_points, px_shape = generate_sample_points_bev(
        sample_range,
        grid_size=sample_grid_size,
        uncertain_label=uct_label)
    sample_points[:, 0] = np.clip(
        sample_points[:, 0],
        a_min=min_x, a_max=max_x)
    sample_points[:, 1] = np.clip(
        sample_points[:, 1],
        a_min=min_y, a_max=max_y)
    px = uct_label.sample_prob(
        sample_points,
        sample_grid=sample_grid_size)
    px /= sample_grid_size**2
    px[px>=1] = 1
    return sample_points, px, px_shape

def get_sample_points_with_probs_3d(
        mean, cov,
        x_range, y_range, z_range,
        sample_range, sample_grid_size):
    '''
    Compute sample points and their probabilities given the mean and cov.
    Args:
        mean (np.ndarray, (1,7))
        cov (np.ndarray, (7,7)/(8,8))
        x_range (Tuple[float]): xmin, xmax
        y_range (Tuple[float]): ymin, ymax
        z_range (Tuple[float]): zmin, zmax
        sample_range (float): sampling range in the hidden space of the generative model.
        sample_grid_size (float): sampling grid size in the hidden space of the generative model.
    Returns:
        sample_points (np.ndarray, (N*dim_z, 3))
        px (np.ndarray, (N*dim_z))
        px_shape (dim_x, dim_y): N = dim_x * dim_y
    '''
    min_x, max_x = x_range
    min_y, max_y = y_range
    min_z, max_z = z_range
    
    uct_label = uncertain_label_3D(mean)
    uct_label.set_posterior(mean, cov)
    sample_points, px_shape = generate_sample_points_3d(
        sample_range,
        grid_size=sample_grid_size,
        uncertain_label=uct_label)
    sample_points[:, 0] = np.clip(
        sample_points[:, 0],
        a_min=min_x, a_max=max_x)
    sample_points[:, 1] = np.clip(
        sample_points[:, 1],
        a_min=min_y, a_max=max_y)
    sample_points[:, 2] = np.clip(
        sample_points[:, 2],
        a_min=min_z, a_max=max_z)
    px = uct_label.sample_prob(
        sample_points,
        sample_grid=sample_grid_size)
    px /= sample_grid_size**2
    px[px>=1] = 1
    return sample_points, px, px_shape

def bbox3d_to_obj(bbox3d:np.ndarray)-> UdiObj:
    '''
    Convert <bbox3d> to <obj>
    Args:
        bbox3d (1,7): [xc, yc, zc, l, w, h, ry],
        where [xc, yc, zc] indicates the center.
    Returns:
        obj: obj.array is [xc, yc, zb, l, w, h, ry],
        where [xc, yc, zb] indicates the bottom center.
    '''
    obj = UdiObj(bbox3d)
    obj.z -= obj.h/2.0
    return obj

def bbox3d_to_kittiobj(bbox3d:np.ndarray, calib:KittiCalib)-> KittiObj:
    '''
    Convert <bbox3d> to <Kittiobj>
    Args:
        bbox3d (1,7): [xc, yc, zc, l, w, h, ry],
        where [xc, yc, zc] indicates the center.
    Returns:
        obj: KittiObj
    '''
    udiobj = bbox3d_to_obj(bbox3d)
    cns_Flidar = udiobj.get_bbox3d_corners()
    cns_Fcam = calib.lidar2leftcam(cns_Flidar)
    kittiobj = KittiObj()
    kittiobj.from_corners(calib=calib,corners=cns_Fcam,
        cls=udiobj.cls,score=1.0)
    kittiobj.score = None
    return kittiobj

def vis_spatial_uncertainty_3d(
    pc: np.ndarray,
    bboxes_3d: np.ndarray,
    x_range: Tuple[float],
    y_range: Tuple[float],
    z_range: Tuple[float],
    grid_size: Tuple[float],
    save_path: str,
    prior_scaler:float=0.5,
    sample_grid_size:float=0.1,
    sample_range:float=3.0):
    '''
    Visualize spatial uncertainty.
    Args:
        pc ([N, 3+]): a frame of point cloud
        bboxes_3d ([M, 7]): 3D bounding boxes. Each box should
            be [xc,yc,zc,l,w,h,ry], where [xc,yc,zc] should be box center
        x_range: (xmin, xmax)
        y_range: (ymin, ymax)
        z_range: (zmin, zmax)
        grid_size: (x_res, y_res, z_res)
        prior_scale: weight of prior distributions in the inference
        sample_grid_size: grid size for generating sample points
        sample_range: sample range for generating sample points
    '''
    # infer predictive distributions
    # and sample points for visualization
    inference = label_inference_3D(
        degree_register=1,
        gen_std=0.25,
        prob_outlier=0.8,
        boundary_sample_interval=0.05)
    obj_list, lp_list, sp_list, px_list = [], [], [], []
    for bbox_3d in bboxes_3d:
        obj = bbox3d_to_obj(bbox_3d)
        pc_idxes = get_points_with_margin(pc, obj, margin=1)
        label_posterior, sample_points, px, px_shape = infer_and_sample(
            inference, pts=pc[pc_idxes], obj=obj,
            x_range=x_range, y_range=y_range, z_range=z_range,
            prior_scaler=prior_scaler,
            grid_size=sample_grid_size,
            sample_range=sample_range)
        obj_list.append(obj)
        lp_list.append(label_posterior)
        sp_list.append(sample_points)
        px_list.append(px)
    # create zbins
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    dx, dy, dz = grid_size
    scale_x = lambda pts_x: (pts_x - xmin ) / dx
    scale_y = lambda pts_y: (ymax - pts_y ) / dy
    zbins = np.arange(zmin, zmax, dz)
    zbins = np.concatenate([np.array([-np.inf]), zbins, np.array([np.inf])])
    n_zbins = zbins.shape[0]
    # visualize for each zbin
    num_cols = 3
    num_rows = n_zbins / num_cols + 1
    plt.figure(figsize=(10*num_cols, 12*num_rows), dpi=100)
    plt.subplots_adjust(None, None, None, None,0,0.2)
    for i, (zmin_, zmax_) in enumerate(zip(zbins[:-1], zbins[1:])):
        pc_idxes = get_points_in_zrange(pc, (zmin_, zmax_))
        if pc_idxes.shape[0] == 0:
            continue
        bevimg = BEVImage(x_range, y_range, (dx, dy))
        bevimg.from_lidar(pc[pc_idxes])
        [bevimg.draw_box(obj, calib=UdiCalib, c='red') for obj in obj_list]

        ax = plt.subplot(num_rows+1, num_cols, i+1)
        ax.imshow(bevimg.data)
        ax.set_title(f"z={zmin_:.2f} to {zmax_:.2f}")
        for lp, sp, px in zip(lp_list, sp_list, px_list):
            sp_idxes = get_points_in_zrange(sp, (zmin_, zmax_))
            if sp_idxes.shape[0] == 0:
                continue
            new_sp = sp[sp_idxes].reshape(*px_shape, -1, 3)[..., 0, :]
            new_px = px[sp_idxes].reshape(*px_shape, -1).sum(axis=-1)
            ax.contour(scale_x(new_sp[..., 0]), scale_y(new_sp[..., 1]),
                new_px, levels=np.array(range(39))/40.0, vmin=0, vmax=0.5, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')

def infer_and_sample(
    inference,
    pts: np.ndarray,
    obj,
    x_range:Tuple[float],
    y_range:Tuple[float],
    z_range:Tuple[float],
    prior_scaler:float=0.5,
    grid_size:float=0.1,
    sample_range:float=3.0)->Tuple[
        uncertain_label_3D, np.ndarray, np.ndarray, Tuple[int, int]]:
    '''
    Inference predictive distribution with the bounding box (<obj>) and its related points (<pts>) using the mean-field variational inference approach.
    Args:
        inference (label_inference_3D): class for infer with the generative model.
        pts (N,3): points related to <obj>.
        obj (UdiObj): bounding box.
        x_range: (min_x, max_x)
        y_range: (min_y, max_y)
        z_range: (min_z, max_z)
        prior_scale: weight of prior distribution in the inference
        grid_size: grid size for generating sample points
        sample_range: sample range for generating sample points
    Returns:
        label_posterior (uncertain_label_3D): predictive distribution of label
        sample_points (np.ndarray (M,3)): sampled points for visualization and JIoU evaluation
        px (np.ndarray (M,)): probabilities of each sample point
        px_shape (Tuple[Int]): x_shape, y_shape
    '''
    min_x, max_x = x_range
    min_y, max_y = y_range
    min_z, max_z = z_range

    label_posterior = inference.infer(
        pts[:, :3],
        np.array([[obj.x, obj.y, obj.z,
                    obj.l, obj.w, obj.h,
                    obj.theta]]),
        prior_scaler=prior_scaler)
    sample_points, px_shape = generate_sample_points_3d(
        sample_range,
        grid_size=grid_size,
        uncertain_label=label_posterior)
    sample_points[:, 0] = np.clip(
        sample_points[:, 0],
        a_min=min_x, a_max=max_x)
    sample_points[:, 1] = np.clip(
        sample_points[:, 1],
        a_min=min_y, a_max=max_y)
    sample_points[:, 2] = np.clip(
        sample_points[:, 2],
        a_min=min_z, a_max=max_z)
    px = label_posterior.sample_prob(
        sample_points,
        sample_grid=grid_size)
    px /= grid_size**2
    px[px>=1] = 1
    return label_posterior, sample_points, px, px_shape

def generate_sample_points_3d(
    range_:float,
    grid_size:float,
    uncertain_label)->Tuple[np.ndarray, np.ndarray]:
    '''
    Generate sample points for the <uncertain_label_3d>
    Args:
        range_: in z space
        grid_size: in z space
        uncertain_label: (uncertain_label_bev)
    Returns:
        sample_points (N,3)
        sample_shape (2,): x_shape, y_shape
    '''
    x = np.arange(-range_/2+grid_size/2, range_/2, grid_size)
    y = x; z = x
    zx, zy, zz = np.meshgrid(x, y, z)
    z_normed = np.concatenate(
        (zx.reshape((-1, 1)),
         zy.reshape((-1, 1)),
         zz.reshape((-1, 1))), axis=1)
    # samples are aligned with label bounding box
    lwh = np.array([uncertain_label.x0[3:6]]).reshape(1,3)
    sample_points = np.matmul(
        z_normed * lwh,
        uncertain_label.rotmat.transpose()) + uncertain_label.x0[0:3]
    return sample_points, np.array((x.shape[0], y.shape[0]))

def vis_spatial_uncertainty_bev(
    pc: np.ndarray,
    bboxes_3d: np.ndarray,
    x_range: Tuple[float],
    y_range: Tuple[float],
    grid_size: Tuple[float],
    save_path: str,
    prior_scaler:float=0.5,
    sample_grid_size:float=0.1,
    sample_range:float=3.0):
    '''
    Visualize spatial uncertainty.
    Args:
        pc ([N, 3+]): a frame of point cloud
        bboxes_3d ([M, 7]): 3D bounding boxes. Each box should
            be [xc,yc,zc,l,w,h,ry], where [xc,yc,zc] should be box center
        x_range: (xmin, xmax)
        y_range: (ymin, ymax)
        grid_size: (x_res, y_res)
        prior_scale: weight of prior distributions in the inference
        sample_grid_size: grid size for generating sample points
        sample_range: sample range for generating sample points
    '''
    inference = label_inference_BEV(
        degree_register=1,
        gen_std=0.25,
        prob_outlier=0.8,
        boundary_sample_interval=0.05)
    
    bevimg = BEVImage(x_range, y_range, grid_size)
    bevimg = bevimg.from_lidar(pc[:, :3], scale=1)
    for bbox_3d in bboxes_3d:
        obj = bbox3d_to_obj(bbox_3d)
        bevimg.draw_box(obj, calib=UdiCalib, c='red')

    plt.figure(figsize=(bevimg.data.shape[1]/100,
        bevimg.data.shape[0]/100), dpi=100)
    plt.imshow(bevimg.data)
    plt.axis("off")
    for bbox_3d in bboxes_3d:
        obj = bbox3d_to_obj(bbox_3d)
        pc_idxes = get_points_with_margin(pc[:, :3], obj, margin=1)
        draw_spatial_uncertainty_bev(
            inference, obj, pc[pc_idxes], bevimg,
            prior_scaler=prior_scaler,
            sample_grid_size=sample_grid_size,
            sample_range=sample_range)
    # bevimg.save(save_path.replace('.', '_bev.'))
    plt.savefig(save_path, bbox_inches='tight')

def draw_spatial_uncertainty_bev(
    inference,
    obj: UdiObj,
    pts: np.ndarray,
    bevimg: BEVImage,
    prior_scaler:float=0.5,
    sample_grid_size:float=0.1,
    sample_range:float=3.0):
    '''
    Args:
        inference: class for implementing Bayesian inference
        obj: Object
        pts (N,3+): pts related to the obj.
        bevimg: BEV image
        prior_scale: weight of prior distributions in the inference
        sample_grid_size: grid size for generating sample points
        sample_range: sample range for generating sample points
    '''
    min_x, max_x = bevimg.x_range
    min_y, max_y = bevimg.y_range
    dx, dy = bevimg.grid_size

    label_posterior = inference.infer(
        pts[:, :2],
        np.array([[obj.x, obj.y, obj.l, obj.w, obj.theta]]),
        prior_scaler=prior_scaler)
    sample_points, px_shape = generate_sample_points_bev(
        sample_range,
        grid_size=sample_grid_size,
        uncertain_label=label_posterior)
    sample_points[:, 0] = np.clip(
        sample_points[:, 0],
        a_min=min_x, a_max=max_x)
    sample_points[:, 1] = np.clip(
        sample_points[:, 1],
        a_min=min_y, a_max=max_y)
    px = label_posterior.sample_prob(
        sample_points,
        sample_grid=sample_grid_size)
    px /= sample_grid_size**2
    px[px>=1] = 1

    scale_x = lambda pts_x: (pts_x - min_x ) / dx
    scale_y = lambda pts_y: (max_y - pts_y ) / dy
    plt.contour(
        scale_x(sample_points[:,0].reshape(px_shape)),
        scale_y(sample_points[:,1].reshape(px_shape)),
        px.reshape(px_shape),
        levels=np.array(range(20))/19.0,
        vmin=0, vmax=1, alpha=0.5)

def generate_sample_points_bev(
    range_:float,
    grid_size:float,
    uncertain_label)->Tuple[np.ndarray, np.ndarray]:
    '''
    Generate sample points for the <uncertain_label>
    Args:
        range_: in z space
        grid_size: in z space
        uncertain_label (uncertain_label_bev)
    Returns:
        sample_points (N,2)
        sample_shape (2,): x_shape, y_shape
    '''
    x = np.arange(
        -range_ / 2 + grid_size / 2,
        range_ / 2, grid_size)
    y = x
    zx, zy = np.meshgrid(x, y)
    z_normed = np.concatenate((zx.reshape((-1, 1)), zy.reshape((-1, 1))), axis=1)
    # samples are aligned with label bounding box
    sample_points = np.matmul(z_normed * uncertain_label.x0[2:4],
        uncertain_label.rotmat.transpose()) + uncertain_label.x0[0:2]
    return sample_points, np.array((x.shape[0], y.shape[0]))

def get_points_with_margin(pts:np.ndarray, obj, margin:float=0, calib=None)->np.ndarray:
    '''
    Get points inside the bounding box defined by <obj>.
    Args:
        pc (N, 3): points
        obj (UdiObj/KittiObj): bounding box
        margin: in meter unit
        calib KittiCalib
    Returns:
        indices (M,): indices of points of <pc>
            enclosed by the bounding box defined by <obj>.
    '''
    crop_obj = deepcopy(obj)
    crop_obj.l += margin
    crop_obj.w += margin
    crop_obj.h += margin
    if isinstance(obj, UdiObj):
        return crop_obj.get_pts_idx(pts)
    elif isinstance(obj, KittiObj):
        return crop_obj.get_pts_idx(pts, calib)
    else:
        raise RuntimeError(f"Unrecognized obj type: {type(obj)}")

def get_points_in_zrange(pts:np.ndarray, z_range:Tuple[float])->np.ndarray:
    '''
    Get points within the <zrange>.
    Args:
        pts (N, 3): points, z axis is the last dim.
        z_range: (zmin, zmax)
    Returns:
        indices (N,): indices of points of <pts>
            within the [zmin, zmax).
    '''
    zmin, zmax = z_range
    return np.nonzero(np.logical_and(
        pts[:, 2] >= zmin,
        pts[:, 2] < zmax))[0]
