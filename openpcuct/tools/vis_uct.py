'''
This script is to visualize predictive distribution of 3D object detection results.
It consumes a pickle file, which should contains all necessary information for visualization,
and render multiple desired images from these data.
'''
import fire
import numpy as np
from pathlib import Path

from det3.ops import read_pkl
from det3.utils.utils import read_image
from det3.dataloader.kittidata import KittiCalib
from pcuct.utils.eval_utils import visualize_bevimg,\
    visualize_fvimg, visualize_bevimg_uct, \
    visualize_bevimg3d_uct, visualize_fvimg_uct
from pcuct.gen_model.label import uncertain_label_BEV, uncertain_label_3D

def parse_data(path, gen_uct_dir=None, data_dir=None):
    '''
    Args:
        path(str): path to the pkl file.
        gen_uct_dir(str): directory contains uncertainty pkl files(generative models). It will be loaded if the "pdist_mean_3d" is missing.
        data_dir(str): directory contains image and calib files.
        It will be loaded if uses "image" and "calib"
    Returns:
        plt_data:{
                "frame_id": str,
                "image": optional,
                "calib": optional,
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
    data = read_pkl(path)
    if "pdist_mean_3d" not in data.keys() and gen_uct_dir is not None:
        # load generative-model-based uncertainty
        gen_uct_path = Path(gen_uct_dir) / f"{data['frame_id']}.pkl"
        uct_data = read_pkl(gen_uct_path)
        keys = ["pdist_mean_bev", "pdist_cov_bev",
                "pdist_mean_3d", "pdist_cov_3d"]
        modify_key_fn = lambda k: k.replace("pdist", "post")
        tmp_dict = {k:[] for k in keys}
        for itm in uct_data:
            k = "pdist_mean_bev"
            v = uncertain_label_BEV.feature0_to_x0(itm[modify_key_fn(k)])
            tmp_dict[k].append(v.reshape(-1, 5))
            k = "pdist_cov_bev"
            v = itm[modify_key_fn(k)]
            v_dim = int(np.sqrt(v.size))
            tmp_dict[k].append(itm[modify_key_fn(k)].reshape(v_dim, v_dim))
            k = "pdist_mean_3d"
            v = uncertain_label_3D.feature0_to_x0(itm[modify_key_fn(k)])
            tmp_dict[k].append(v.reshape(-1, 7))
            k = "pdist_cov_3d"
            v = itm[modify_key_fn(k)]
            v_dim = int(np.sqrt(v.size))
            tmp_dict[k].append(itm[modify_key_fn(k)].reshape(v_dim, v_dim))
        for k, v in tmp_dict.items():
            if k in ["pdist_cov_bev", "pdist_cov_3d"]:
                continue
            tmp_dict[k] = np.concatenate(v, axis=0)
        data.update(tmp_dict)
    if data_dir is not None:
        image = read_image(Path(data_dir)/ "image_2" / f"{data['frame_id']}.png")
        calib = KittiCalib(Path(data_dir)/ "calib" / f"{data['frame_id']}.txt")\
            .read_calib_file()
        data.update({
            "image": image,
            "calib": calib
        })
    return data

def setup_dir(save_dir, frame_id, surfix=""):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    surfix = "" if len(surfix) == 0 else f"_{surfix}"
    return Path(save_dir)/f"{frame_id}{surfix}.png"

def render_bevimg(
    plt_data,
    x_range,
    y_range,
    grid_size,
    bool_gt):
    return visualize_bevimg(plt_data,
        x_range, y_range, grid_size, bool_gt)

def render_bevimg_uct(
    plt_data,
    x_range,
    y_range,
    grid_size,
    sample_grid_size,
    sample_range,
    save_path,
    bool_gt=True):
    visualize_bevimg_uct(plt_data,
        x_range, y_range, grid_size,
        sample_range, sample_grid_size, save_path, bool_gt=bool_gt)

def render_fvimg(
    plt_data,
    bool_gt,
    bool_image):
    return visualize_fvimg(plt_data, bool_gt, bool_image)

def render_fvimg_uct(
    plt_data,
    x_range,
    y_range,
    z_range,
    sample_range,
    sample_grid_size,
    save_path,
    bool_gt,
    bool_image):
    return visualize_fvimg_uct(plt_data,
        x_range, y_range, z_range,
        sample_range, sample_grid_size, save_path,
        bool_gt=bool_gt, bool_image=bool_image)

def render_hbevimg_uct(
    plt_data,
    x_range,
    y_range,
    z_range,
    grid_size,
    sample_grid_size,
    sample_range,
    save_path):
    visualize_bevimg3d_uct(plt_data,
        x_range, y_range, z_range, grid_size,
        sample_range, sample_grid_size, save_path)

class Visualizer:
    def render_bevimg(
        self,
        path, save_dir,
        x_range, y_range, grid_size, bool_gt):
        plt_data = parse_data(path)
        save_path = setup_dir(save_dir, plt_data["frame_id"], surfix="bevimg")
        bevimg = render_bevimg(
            plt_data,
            x_range,
            y_range,
            grid_size,
            bool_gt=bool_gt)
        bevimg.save(save_path)

    def render_bevimg_uct(self,
        path, save_dir,
        x_range, y_range, grid_size,
        sample_grid_size, sample_range, bool_gt=True, gen_uct_dir=None):
        plt_data = parse_data(path, gen_uct_dir=gen_uct_dir)
        save_path = setup_dir(save_dir, plt_data["frame_id"], surfix="bevimguct")
        render_bevimg_uct(
            plt_data, x_range, y_range, grid_size,
            sample_grid_size, sample_range, save_path, bool_gt=bool_gt)

    def render_fvimg(self,
        path,
        save_dir,
        bool_gt,
        bool_image,
        data_dir=None):
        plt_data = parse_data(path, data_dir=data_dir)
        save_path = setup_dir(save_dir, plt_data["frame_id"], surfix="fvimg")
        fvimg = render_fvimg(plt_data, bool_gt=bool_gt, bool_image=bool_image)
        fvimg.save(save_path)

    def render_fvimg_uct(self,
        path,
        save_dir,
        x_range, y_range, z_range,
        sample_range, sample_grid_size,
        bool_gt,
        bool_image,
        gen_uct_dir=None,
        data_dir=None):
        plt_data = parse_data(path, gen_uct_dir=gen_uct_dir, data_dir=data_dir)
        save_path = setup_dir(save_dir, plt_data["frame_id"], surfix="fvimguct")
        render_fvimg_uct(plt_data,
            x_range, y_range, z_range,
            sample_range, sample_grid_size, save_path, bool_gt, bool_image)

    def render_hbevimg_uct(self,
        path, save_dir,
        x_range, y_range, z_range, grid_size,
        sample_grid_size, sample_range):
        plt_data = parse_data(path)
        save_path = setup_dir(save_dir, plt_data["frame_id"], surfix="hbevimguct")
        render_hbevimg_uct(
            plt_data,
            x_range, y_range, z_range, grid_size,
            sample_grid_size, sample_range, save_path)

if __name__ == "__main__":
    fire.Fire(Visualizer)
