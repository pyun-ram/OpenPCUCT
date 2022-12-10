import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from det3.ops import read_txt, write_pkl
from det3.dataloader.kittidata import KittiData, KittiLabel
from pcuct.gen_model.infer import label_inference_BEV, label_inference_3D
from pcuct.gen_model.vis import get_points_with_margin

def clip_ry(ry):
    '''
    clip_ry into [-pi/2, pi/2]
    '''
    while ry < -np.pi/2.0:
        ry += np.pi
    while ry > np.pi/2.0:
        ry -= np.pi
    assert -np.pi/2.0 <= ry <= np.pi/2.0
    return ry

def generate_uncertainty(
    data_dir:str,
    anno_dir:str,
    index:str,
    prior_scaler=0.5)->List[Dict[str, np.ndarray]]:
    '''
    Inference predictive distributions with a generative model.
    Args:
        data_dir: data dir of the KITTI dataset
        anno_dir: the folder contains the labels/detections
        index: index of the data (e.g. "000001")
        prior_scaler: prior_scale: weight of prior distributions in the inference
    Returns:
        pdist_list: it contains predictive distribution (BEV&3D) for each object.
            each dictionary contains:{
                "post_mean_bev": np.ndarray (5,)/(6,),
                "post_cov_bev": np.ndarray (5,5)/(6,6),
                "post_mean_3d": post_mean_3d (7,)/(8,),
                "post_cov_3d": post_cov_3d (7,7)/(8,8)}
    '''
    # read data
    calib, _, _, pc = KittiData(
        root_dir=data_dir, idx=index,
        output_dict={
            "calib": True,
            "image": False,
            "label": False,
            "velodyne": True}).read_data()
    label = KittiLabel(Path(anno_dir)/f"{index}.txt")\
        .read_label_file(no_dontcare=False)

    inference_bev = label_inference_BEV(
        degree_register=1,
        gen_std=0.25,
        prob_outlier=0.8,
        boundary_sample_interval=0.05)
    inference_3d = label_inference_3D(
        degree_register=1,
        gen_std=0.25,
        prob_outlier=0.8,
        boundary_sample_interval=0.05)
    pdist_list = []
    for obj in label.data:
        pc_idx = get_points_with_margin(pc[:, :3], obj,
            margin=1.0, calib=calib)
        pts_obj = pc[pc_idx]
        # change obj in Fcam to Flidar
        # -change center
        # -change ry
        bottom_Fcam = np.array([[obj.x, obj.y, obj.z]])
        bottom_Flidar = calib.leftcam2lidar(bottom_Fcam)
        bbox3d = np.array([
            bottom_Flidar[0,0],
            bottom_Flidar[0,1],
            bottom_Flidar[0,2]+obj.h/2.0,
            obj.l, obj.w, obj.h,
            clip_ry(np.pi/2-obj.ry)]).reshape(1,7)
        # infer
        label_posterior_bev = inference_bev.infer(
            pts_obj[:, :2], bbox3d[0, [0,1,3,4,6]],
            prior_scaler=prior_scaler)
        label_posterior_3d = inference_3d.infer(
            pts_obj[:, :3], bbox3d,
            prior_scaler=prior_scaler)
        # save
        post_mean_bev, post_cov_bev = label_posterior_bev.posterior
        post_mean_3d, post_cov_3d = label_posterior_3d.posterior
        pdist = {
            "post_mean_bev": post_mean_bev,
            "post_cov_bev": post_cov_bev,
            "post_mean_3d": post_mean_3d,
            "post_cov_3d": post_cov_3d,
        }
        pdist_list.append(pdist)
    return pdist_list

def main(data_dir, anno_dir):
    index_file_path = Path(data_dir)/"../ImageSets/val.txt"
    save_dir = Path(anno_dir)/"../uncertainty"
    assert not save_dir.exists(), f"save_dir exists"
    save_dir.mkdir()
    indices = read_txt(index_file_path)
    for index in tqdm(indices):
        uct = generate_uncertainty(data_dir, anno_dir, index, prior_scaler=0.5)
        write_pkl(uct, Path(save_dir)/f"{index}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the directory containing kitti data.')
    parser.add_argument('--anno_dir', type=str, default=None, help='Path to the directory container annotations.')
    args = parser.parse_args()
    main(args.data_dir, args.anno_dir)
