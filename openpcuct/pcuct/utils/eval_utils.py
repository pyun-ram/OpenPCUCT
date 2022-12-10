import pickle
import time
from det3.dataloader.kittidata import KittiCalib

import torch
import tqdm
import numpy as np
import scipy
from pathlib import Path
from matplotlib import pyplot as plt

from det3.ops import write_pkl
from det3.visualizer.vis import BEVImage, FVImage
from det3.dataloader.udidata import UdiCalib, UdiFrame
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcuct.gen_model.vis import get_sample_points_with_probs_bev, \
    get_sample_points_with_probs_3d, bbox3d_to_obj, get_points_in_zrange, bbox3d_to_kittiobj

cls2color = {
    "car": "yellow",
    "truck": "orange",
    "construction_vehicle": "green",
    "bus": "blue",
    "trailer": "brown",
    "barrier": "pink",
    "motorcycle": "violet",
    "bicycle": "red",
    "pedestrian": "cyan",
    "traffic_cone": "white",
    "cyclist": "red",
    "forklift": "green",
    "golf_car": "brown",
    "unknown": "grey",
    "Car": "yellow",
    "Pedestrian": "cyan",
    "Cyclist": "red",
    "Van": "blue",
    "Truck": "orange",
    "Unknown": "grey"
}

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def save_plt_data(batch_dict, pred_dicts, save_dir, idx2cls):
    '''
    Save necessary data for plotting.
    Args:
        batch_dict(Dict)
        pred_dicts(List[Dict])
        save_dir(str)
        idx2cls(Dict[int, str]): converting class index to class name (1-indexed)
        e.g. {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
    '''
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, pred_dict in enumerate(pred_dicts):
        plt_data = parse_batch_dict(batch_dict, index=i, idx2cls=idx2cls)
        plt_data.update(parse_pred_dict(pred_dict, idx2cls=idx2cls))
        write_pkl(plt_data, save_dir/f"{plt_data['frame_id']}.pkl")

def eval_visualize(
    batch_dict,
    pred_dicts,
    save_dir,
    x_range,
    y_range,
    z_range,
    idx2cls):
    '''
    Visualize detection results
    Args:
        batch_dict(Dict)
        pred_dicts(List[Dict])
        save_dir(str)
        x_range(Tuple[float]): x_min, x_max
        y_range(Tuple[float]): y_min, y_max
        z_range(Tuple[float]): z_min, z_max
        idx2cls(Dict[int,str])
    Note: plt_data = {
        "frame_id": str,
        "points": np.ndarray(O, 3),
        "pred_boxes": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry]
        "pred_labels": List[str] (N),
        "pred_scores": np.ndarray(N),
        "gt_boxes": np.ndarray(M, 7), [xc,yc,zc,l,w,h,ry]
        "gt_labels": List[str] (M),
        "pdist_mean_bev": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry]
        "pdist_cov_bev": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry] std
        "pdist_mean_3d": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry]
        "pdist_cov_3d": np.ndarray(N,7), [xc,yc,zc,l,w,h,ry] std
    }
    '''
    bev_dir = Path(save_dir)/"bevimg"
    fv_dir = Path(save_dir)/"fvimg"
    bevuct_dir = Path(save_dir)/"bevimg_uct"
    d3uct_dir = Path(save_dir)/"d3_uct"
    [itm.mkdir(parents=True, exist_ok=True) 
        for itm in [bev_dir, fv_dir, bevuct_dir, d3uct_dir]]
    # parse batch_dict, pred_dicts
    plt_data_dicts = []
    for i, pred_dict in \
        enumerate(pred_dicts):
        plt_data = parse_batch_dict(batch_dict, index=i, idx2cls=idx2cls)
        plt_data.update(parse_pred_dict(pred_dict, idx2cls=idx2cls))
        plt_data_dicts.append(plt_data)
    # visualize&save detection results (BEV/FV)
    for plt_data in plt_data_dicts:
        frame_id = plt_data['frame_id']
        visualize_bevimg(plt_data,
            x_range, y_range, grid_size=(0.05,0.05), bool_gt=True)\
            .save(f"{bev_dir/frame_id}.png")
        visualize_fvimg(plt_data, bool_gt=True).save(f"{fv_dir/frame_id}.png")
    # visualize&save predictive dirstribution (BEV/3D)
    for plt_data in plt_data_dicts:
        frame_id = plt_data['frame_id']
        visualize_bevimg_uct(plt_data,
            x_range, y_range, grid_size=(0.05, 0.05), sample_range=3.0,
            sample_grid_size=0.1, save_path=f"{bevuct_dir/frame_id}.png")
        visualize_fvimg_uct(plt_data,
            x_range, y_range, z_range, sample_range=3.0, sample_grid_size=0.1, save_path=f"{d3uct_dir/frame_id}.png", bool_image=False, bool_gt=True)

def parse_batch_dict(batch_dict, index, idx2cls):
    '''
    Parse plot_data from <batch_dict> of <index>
    Args:
        batch_dict (Dict):
        index (int): batch index
        idx2cls (Dict[int,str])
    Returns:
        plt_data: {
        "frame_id": str,
        "points": np.ndarray (N, 3),
        "gt_boxes": np.ndarray(M, 7), [xc,yc,zc,l,w,h,ry]
        "gt_labels": List[str] (M),
        }
    '''
    frame_id = batch_dict['frame_id'][index]
    points = batch_dict['points'][:, 1:4]
    points = points[batch_dict["points"][:, 0] == index]
    gt_boxes = batch_dict["gt_boxes"][index, :, :7]
    gt_labels = batch_dict["gt_boxes"][index, :, -1].cpu().numpy()
    gt_labels = [idx2cls[int(itm)] if itm != 0 else "?" for itm in gt_labels]
    return {
        "frame_id": frame_id,
        "points": points.cpu().numpy(),
        "gt_boxes": gt_boxes.cpu().numpy(),
        "gt_labels": gt_labels
    }

def parse_pred_dict(pred_dict, idx2cls):
    '''
    Parse plot_data from pred_dict.
    Args:
        pred_dict (Dict):
        idx2cls (Dict[int:str]): {idx: cls}
    Returns:
        plt_data: {
        "pred_boxes": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry]
        "pred_labels": List[str] (N),
        "pred_scores": np.ndarray(N),
        "pdist_mean_bev": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry]
        "pdist_cov_bev": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry] std
        "pdist_mean_3d": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry]
        "pdist_cov_3d": np.ndarray(N,7), [xc,yc,zc,l,w,h,ry] std
        }
    '''
    pred_labels = pred_dict['pred_labels'].cpu().numpy()
    pred_labels = [idx2cls[int(itm)] for itm in pred_labels]
    plt_data = {
        "pred_boxes": pred_dict['pred_boxes'].cpu().numpy(),
        "pred_labels": pred_labels,
        "pred_scores": pred_dict['pred_scores'].cpu().numpy()
        }
    if pred_dict.get("pdist_boxes_mean", None) is not None:
        pdist_boxes_mean = pred_dict['pdist_boxes_mean'].cpu().numpy()
        pdist_boxes_cov = pred_dict['pdist_boxes_cov'].cpu().numpy()
        if len(pdist_boxes_cov.shape) == 3:
            # if it is a cov matrix, extract the cov matrix of bev
            assert pdist_boxes_cov.shape[-1] == pdist_boxes_cov.shape[-2]
            pdist_cov_3d = pdist_boxes_cov
            pdist_cov_bev = np.concatenate(
                [pdist_cov_3d[:, i, [0,1,3,4,6]].reshape(-1, 1, 5)
                for i in [0,1,3,4,6]], axis=1)
        elif len(pdist_boxes_cov.shape) == 2:
            # if it is a diagonal cov vector, extract the cov vector of bev
            pdist_cov_3d = pdist_boxes_cov
            pdist_cov_bev = pdist_cov_3d[:, 0,1,3,4,6]
        else:
            raise NotImplementedError
        plt_data.update({
            "pdist_mean_bev": pdist_boxes_mean[:, [0,1,3,4,6]],
            "pdist_cov_bev": pdist_cov_bev,
            "pdist_mean_3d": pdist_boxes_mean,
            "pdist_cov_3d": pdist_cov_3d
        })
    return plt_data

def visualize_bevimg(
    plt_data,
    x_range,
    y_range,
    grid_size,
    bool_gt):
    '''
    Args:
        plt_data: {
            "points": np.ndarray(L, 3),
            "gt_boxes": np.ndarray(M, 7), [xc,yc,zc,l,w,h,ry]
            "gt_labels": List[str] (M),
            "pred_boxes": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry]
            "pred_labels": List[str] (N),
            "pred_scores": np.ndarray(N),
            }
        x_range(Tuple[float]): x_min, x_max
        y_range(Tuple[float]): y_min, y_max
        grid_size(Tuple[float]): dx, dy
        bool_gt(bool): True, if plot ground-truth boxes
    Returns:
        bevimg(BEVImage)
    '''
    bevimg = BEVImage(
        x_range=x_range,
        y_range=y_range,
        grid_size=grid_size)
    bevimg.from_lidar(plt_data["points"])
    if bool_gt:
        gt_labels = plt_data["gt_labels"]
        gt_boxes = plt_data["gt_boxes"]
        for gt_label, gt_box in zip(gt_labels, gt_boxes):
            obj = bbox3d_to_obj(gt_box)
            obj.cls = gt_label
            bevimg.draw_box(obj, calib=UdiCalib, bool_gt=True, width=3)
    labels = plt_data["pred_labels"]
    boxes = plt_data["pred_boxes"]
    scores = plt_data["pred_scores"]
    for label, box, score in zip(labels, boxes, scores):
        obj = bbox3d_to_obj(box)
        obj.cls = label; obj.score = score
        bevimg.draw_box(obj, calib=UdiCalib, bool_gt=False, width=2,
            c=cls2color[obj.cls])
    return bevimg

def default_calib():
    '''
    Returns a default UDICalib class.
    '''
    calib = UdiCalib(None)
    calib._data = {}
    T = np.eye(4).astype(np.float32)
    tx, ty, tz = 20, 0, 0
    T[:3, -1] = tx, ty, tz
    calib.vcam_T = T
    calib._data[(UdiFrame("BASE"), UdiFrame("BASE"))] = \
        np.eye(4).astype(np.float32)
    return calib

def visualize_fvimg(plt_data, bool_gt, bool_image=False):
    '''
    Args:
        plt_data: {
            "points": np.ndarray(L, 3),
            "gt_boxes": np.ndarray(M, 7), [xc,yc,zc,l,w,h,ry]
            "gt_labels": List[str] (M),
            "pred_boxes": np.ndarray(N, 7), [xc,yc,zc,l,w,h,ry]
            "pred_labels": List[str] (N),
            "pred_scores": np.ndarray(N),
            }
        bool_gt(bool): True, if plot ground-truth boxes
    Returns:
        fvimg(FVImage)
    '''
    fvimg = FVImage()
    if not bool_image:
        calib = default_calib()
        fvimg.from_lidar(calib, plt_data["points"], scale=2)
        bbox3d_to_obj_fn = bbox3d_to_obj
    else:
        calib = plt_data["calib"]
        fvimg.from_image(plt_data["image"])
        # Uncomment this line, if you want to draw on lidar points.
        # fvimg.from_lidar(calib, plt_data["points"], scale=2)
        bbox3d_to_obj_fn = lambda bbox3d: bbox3d_to_kittiobj(bbox3d, calib)
    if bool_gt:
        gt_labels = plt_data["gt_labels"]
        gt_boxes = plt_data["gt_boxes"]
        for gt_label, gt_box in zip(gt_labels, gt_boxes):
            obj = bbox3d_to_obj_fn(gt_box); obj.cls = gt_label
            fvimg.draw_3dbox(obj, calib, bool_gt=True, width=3)
    labels = plt_data["pred_labels"]
    boxes = plt_data["pred_boxes"]
    scores = plt_data["pred_scores"]
    for label, box, score in zip(labels, boxes, scores):
        obj = bbox3d_to_obj_fn(box)
        obj.cls = label; obj.score = score
        fvimg.draw_3dbox(obj, calib,
            bool_gt=False, width=2, c=cls2color[obj.cls])
    return fvimg

def visualize_bevimg_uct(
    plt_data,
    x_range,
    y_range,
    grid_size,
    sample_range,
    sample_grid_size,
    save_path,
    bool_gt=True):
    '''
    Args:
        plt_data: {
            "points": np.ndarray(L, 3),
            "pdist_mean_bev": np.ndarray(N,5) [xc,yc,l,w,ry]
            "pdist_cov_bev":
                np.ndarray(N,5)/np.ndarray(N,5,5):[xc,yc,l,w,ry]
                np.ndarray(N,6)/np.ndarray(N,6,6):
                [xc, yc, lcosry, lsinry, wcosry, wsinry]
                }
        x_range(Tuple[float]): x_min, x_max
        y_range(Tuple[float]): y_min, y_max
        grid_size(Tuple[float]): dx, dy
        sample_range(float): sampling range in the hidden space of the generative model.
        sample_grid_size(float): sampling grid size in the hidden space of the generative model.
        save_path(str)
    '''
    bevimg = visualize_bevimg(
        plt_data, x_range, y_range, grid_size, bool_gt=bool_gt)
    plt.figure(figsize=(bevimg.data.shape[1]/100,
        bevimg.data.shape[0]/100), dpi=100)
    plt.imshow(bevimg.data)
    plt.axis("off")
    means = plt_data['pdist_mean_bev']
    covs = plt_data['pdist_cov_bev']
    min_x = x_range[0]
    max_y = y_range[1]
    dx, dy = grid_size
    scale_x = lambda pts_x: (pts_x - min_x ) / dx
    scale_y = lambda pts_y: (max_y - pts_y ) / dy

    for mean, cov in zip(means, covs):
        mean = mean.reshape(-1)
        if cov.size == mean.size:
            cov = np.diag(cov)
        elif cov.size == mean.size ** 2:
            pass
        elif cov.size == 6 * 6 and mean.size == 5:
            pass
        else:
            err_msg = f"Wrong shape of cov: {cov.shape}"
            raise RuntimeError(err_msg)
        sp, px, px_shape = get_sample_points_with_probs_bev(
            mean, cov,
            x_range, y_range,
            sample_range, sample_grid_size)
        plt.contour(
            scale_x(sp[:, 0].reshape(px_shape)),
            scale_y(sp[:, 1].reshape(px_shape)),
            px.reshape(px_shape),
            levels=np.array(range(20))/19.0,
            vmin=0, vmax=1, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')

def visualize_bevimg3d_uct(
    plt_data,
    x_range,
    y_range,
    z_range,
    grid_size,
    sample_range,
    sample_grid_size,
    save_path):
    '''
    Args:
        plt_data: {
            "points": np.ndarray(L, 3),
            "pdist_mean_bev": np.ndarray(N,7) [xc,yc,zc,l,w,h,ry]
            "pdist_cov_bev":
                np.ndarray(N,7)/np.ndarray(N,7,7): [xc,yc,zc,l,w,h,ry]
                np.ndarray(N,8)/np.ndarray(N,8,8):
                [xc, yc, zc, lcosry, lsinry, wcosry, wsinry, h]
                }
        x_range(Tuple[float]): x_min, x_max
        y_range(Tuple[float]): y_min, y_max
        z_range(Tuple[float]): z_min, z_max
        grid_size(Tuple[float]): dx, dy
        sample_range(float): sampling range in the hidden space of the generative model.
        sample_grid_size(float): sampling grid size in the hidden space of the generative model.
        save_path(str)
    '''
    means = plt_data['pdist_mean_3d']
    covs = plt_data['pdist_cov_3d']
    pc = plt_data['points']
    dx, dy, dz = grid_size

    obj_list, sp_list, px_list = [], [], []
    for mean, cov in zip(means, covs):
        mean = mean.reshape(-1)
        if cov.size == mean.size:
            cov = np.diag(cov)
        elif cov.size == mean.size ** 2:
            pass
        else:
            err_msg = f"Wrong shape of cov: {cov.shape}"
            raise RuntimeError(err_msg)
        sp, px, px_shape = get_sample_points_with_probs_3d(
            mean, cov,
            x_range, y_range, z_range,
            sample_range, sample_grid_size)
        obj = bbox3d_to_obj(mean)
        obj_list.append(obj)
        sp_list.append(sp)
        px_list.append(px)
    # create zbins
    xmin, _ = x_range
    _, ymax = y_range
    zmin, zmax = z_range
    scale_x = lambda pts_x: (pts_x - xmin ) / dx
    scale_y = lambda pts_y: (ymax - pts_y ) / dy
    zbins = np.arange(zmin, zmax, dz)
    zbins = np.concatenate([np.array([-np.inf]), zbins, np.array([np.inf])])
    n_zbins = zbins.shape[0]
    # visualize for each zbin
    num_cols = 3
    num_rows = int(n_zbins / num_cols + 1)
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
        for sp, px in zip(sp_list, px_list):
            sp_idxes = get_points_in_zrange(sp, (zmin_, zmax_))
            if sp_idxes.shape[0] == 0:
                continue
            new_sp = sp[sp_idxes].reshape(*px_shape, -1, 3)[..., 0, :]
            new_px = px[sp_idxes].reshape(*px_shape, -1).sum(axis=-1)
            ax.contour(scale_x(new_sp[..., 0]), scale_y(new_sp[..., 1]),
                new_px, levels=np.array(range(39))/40.0, vmin=0, vmax=0.5, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')

def visualize_fvimg_uct(
    plt_data,
    x_range,
    y_range,
    z_range,
    sample_range,
    sample_grid_size,
    save_path,
    bool_gt,
    bool_image=False):
    '''
    Args:
        plt_data: {
            "points": np.ndarray(L, 3),
            "pdist_mean_bev": np.ndarray(N,7) [xc,yc,zc,l,w,h,ry]
            "pdist_cov_bev":
                np.ndarray(N,7)/np.ndarray(N,7,7): [xc,yc,zc,l,w,h,ry]
                np.ndarray(N,8)/np.ndarray(N,8,8):
                [xc, yc, zc, lcosry, lsinry, wcosry, wsinry, h]
                }
        x_range(Tuple[float]): x_min, x_max
        y_range(Tuple[float]): y_min, y_max
        z_range(Tuple[float]): z_min, z_max
        grid_size(Tuple[float]): dx, dy
        sample_range(float): sampling range in the hidden space of the generative model.
        sample_grid_size(float): sampling grid size in the hidden space of the generative model.
        save_path(str)
    '''
    means = plt_data['pdist_mean_3d']
    covs = plt_data['pdist_cov_3d']
    calib = plt_data['calib'] if bool_image else default_calib()

    sp_list, px_list = [], []
    for mean, cov in zip(means, covs):
        mean = mean.reshape(-1)
        if cov.size == mean.size:
            cov = np.diag(cov)
        elif cov.size == mean.size ** 2:
            pass
        elif cov.size == 8*8 and mean.size == 7:
            pass
        else:
            err_msg = f"Wrong shape of cov: {cov.shape}"
            raise RuntimeError(err_msg)
        sp, px, _ = get_sample_points_with_probs_3d(
            mean, cov,
            x_range, y_range, z_range,
            sample_range, sample_grid_size)
        sp_list.append(sp)
        px_list.append(px)
    # visualize for each zbin
    fvimg = visualize_fvimg(plt_data, bool_gt=bool_gt, bool_image=bool_image)
    plt.figure(figsize=(fvimg.data.shape[1]/100,
        fvimg.data.shape[0]/100), dpi=100)
    plt.axis("off")
    plt.imshow(fvimg.data)
    data = np.zeros((fvimg.data.shape[0], fvimg.data.shape[1]))
    z = np.zeros((fvimg.data.shape[0], fvimg.data.shape[1]))
    scale = 1
    err_msg = "Scale can only be 1."
    # Otherwise, it will cause wrong visiualization in far objects.
    assert scale == 1, err_msg
    for sp, px in zip(sp_list, px_list):
        if isinstance(calib, KittiCalib):
            sp_Fcam = calib.lidar2leftcam(sp)
            sp_Fimg = calib.leftcam2imgplane(sp_Fcam)
        elif isinstance(calib, UdiCalib):
            sp_Fcam = calib.transform(sp, source_frame=UdiFrame("Base"), target_frame=UdiFrame("VCAM"))
            sp_Fimg = calib.vcam2imgplane(sp_Fcam)
        for (x, y), (x_, y_, z_), px_ in zip(sp_Fimg, sp_Fcam, px):
            x, y = int(x), int(y)
            d = np.sqrt(x_**2 + y_**2 + z_**2)
            if x < 0 or x >= fvimg.data.shape[1]-scale:
                continue
            if y < 0 or y >= fvimg.data.shape[0]-scale:
                continue
            if z[y, x] == 0 or d < z[y, x]:
                data[y:y+scale, x:x+scale] = px_
                z[y:y+scale, x:x+scale] = d

    data = scipy.ndimage.filters.gaussian_filter(data, 4)
    data /= data.max()
    plt.contour(range(data.shape[1]), range(data.shape[0]), data,
        vmin=0, vmax=1.0,
        levels=np.array(range(1, 9))/10, alpha=1)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)

def eval_one_epoch_uct(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, wdist_dict={}, num_MC_samples=10, pdist_prior={}, bayesian_inference_mode="full-net", dropout_rate=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    final_uct_dir = result_dir / 'final_result' / 'uncertainty'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_uct_dir.mkdir(parents=True, exist_ok=True)
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    det_pdists = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model.bayesian_inference(
                batch_dict,
                wdist_dict,
                num_MC_samples=num_MC_samples,
                pdist_prior=pdist_prior,
                mode=bayesian_inference_mode,
                dropout_rate=dropout_rate)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        pdist = dataset.generate_predictive_distribution(
            batch_dict, pred_dicts, class_names,
            output_path=final_uct_dir if save_to_file else None
        )
        idx2cls={idx+1: cls for idx, cls in enumerate(class_names)}
        save_plt_data(batch_dict, pred_dicts,
            save_dir=result_dir/f"vis_data-{epoch_id}", idx2cls=idx2cls)
        if i < 3:
            pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
            x_range = (pc_range[0], pc_range[3])
            y_range = (pc_range[1], pc_range[4])
            z_range = (pc_range[2], pc_range[5])
            eval_visualize(
                batch_dict, pred_dicts,
                save_dir=result_dir/f"vis-{epoch_id}",
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                idx2cls=idx2cls)
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict
